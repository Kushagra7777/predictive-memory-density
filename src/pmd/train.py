from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import numpy as np
import torch

from .generator import episode,pad_histories
from .io import append_jsonl,load_torch,save_torch,set_seed,synchronize,write_json,progress
from .transformer import Transformer,shifted_loss


def get_batch(rng,cfg,size,split='train',fixed=None):
    histories = fixed if fixed is not None else [episode(rng,split=split,max_context=cfg['model']['context'],reliability=cfg['generator']['reliability']) for _ in range(size)]
    return pad_histories(histories,cfg['model']['context'])


@torch.no_grad()
def validation_loss(model,episodes,device,batch_size=128):
    model.eval()
    total,count = 0.,0
    for start in range(0,len(episodes),batch_size):
        ids,mask = pad_histories(episodes[start:start+batch_size],model.config['context'])
        ids,mask = torch.as_tensor(ids,device=device),torch.as_tensor(mask,device=device)
        loss,n = shifted_loss(model(ids,mask),ids,mask)
        total += loss.item()*n
        count += n
    return total/max(count,1)


def train_transformer(cfg,run,seed,device,smoke=False,resume=False):
    set_seed(seed)
    train_cfg = copy.deepcopy(cfg['train'])
    if smoke:
        train_cfg.update(cfg['smoke'])
    directory = Path(run)/f'seed_{seed}'/('smoke' if smoke else 'teacher')
    directory.mkdir(parents=True,exist_ok=True)
    best_path,last_path = directory/'best.pt',directory/'last.pt'
    if resume and (directory/'training_summary.json').exists() and last_path.exists():
        if load_torch(last_path,'cpu')['step']>=train_cfg['steps']:
            return best_path
    if last_path.exists() and not resume:
        raise FileExistsError(f'{last_path} already exists. Use --resume or a new run directory.')
    model = Transformer(cfg['model']).to(device)
    opt = torch.optim.AdamW(model.optimizer_groups(train_cfg['weight_decay']),
                           lr=train_cfg['learning_rate'],betas=tuple(train_cfg['betas']),eps=train_cfg['eps'])
    rng = np.random.default_rng(seed+10000)
    val_rng = np.random.default_rng(seed+20000)
    val = [episode(val_rng,split='teacher_val',max_context=64) for _ in range(train_cfg['validation_episodes'])]
    begin,best,tokens_seen,initial_loss = 0,float('inf'),0,None
    if resume and last_path.exists():
        ck = load_torch(last_path,device)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['optimizer'])
        rng.bit_generator.state = ck['data_rng']
        torch.set_rng_state(ck['torch_rng'].cpu())
        if device.type=='cuda' and ck.get('cuda_rng'):
            torch.cuda.set_rng_state_all(ck['cuda_rng'])
        begin,best,tokens_seen = ck['step'],ck['best_validation'],ck['tokens_seen']
        initial_loss = ck['initial_loss']
    use_amp = device.type=='cuda' and train_cfg['bf16'] and torch.cuda.is_bf16_supported()
    start_time = time.perf_counter()
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats(device)
    durations = []
    for step in range(begin,train_cfg['steps']):
        tick = time.perf_counter()
        model.train()
        warm = train_cfg['warmup']
        if step < warm:
            lr = train_cfg['learning_rate']*(step+1)/max(warm,1)
        else:
            f = (step-warm)/max(train_cfg['steps']-warm-1,1)
            lr = train_cfg['min_learning_rate']+(train_cfg['learning_rate']-train_cfg['min_learning_rate'])*.5*(1+math.cos(math.pi*f))
        for group in opt.param_groups:
            group['lr'] = lr
        ids,mask = get_batch(rng,cfg,train_cfg['batch_size'])
        ids,mask = torch.as_tensor(ids,device=device),torch.as_tensor(mask,device=device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=use_amp):
            loss,n = shifted_loss(model(ids,mask),ids,mask)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite transformer training loss.')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),train_cfg['clip'])
        opt.step()
        tokens_seen += n
        initial_loss = loss.item() if initial_loss is None else initial_loss
        if len(durations)<100:
            synchronize(device)
            durations.append(time.perf_counter()-tick)
        append_jsonl(directory/'train.jsonl',dict(step=step+1,loss=loss.item(),learning_rate=lr,tokens_seen=tokens_seen))
        validate = (step+1)%train_cfg['validation_every']==0 or step+1==train_cfg['steps']
        if validate:
            val_loss = validation_loss(model,val,device)
            improve = val_loss < best
            best = min(best,val_loss)
            ck = dict(model=model.state_dict(),optimizer=opt.state_dict(),model_config=cfg['model'],seed=seed,
                      step=step+1,best_validation=best,tokens_seen=tokens_seen,data_rng=rng.bit_generator.state,
                      torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
                      initial_loss=initial_loss,profile=cfg['profile'],smoke=smoke)
            save_torch(last_path,ck)
            if improve:
                save_torch(best_path,ck)
            append_jsonl(directory/'validation.jsonl',dict(step=step+1,loss=val_loss,best=best))
            progress(f'seed={seed} teacher step={step+1} train={loss.item():.5f} validation={val_loss:.5f}')
    synchronize(device)
    write_json(directory/'training_summary.json',dict(seed=seed,smoke=smoke,profile=cfg['profile'],best_validation=best,
               initial_loss=initial_loss,final_loss=loss.item() if begin<train_cfg['steps'] else None,
               unmasked_targets=tokens_seen,padded_token_slots=train_cfg['steps']*train_cfg['batch_size']*64,
               elapsed_seconds=time.perf_counter()-start_time,first_100_step_seconds=durations,
               peak_vram_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
               parameter_count=sum(p.numel() for p in model.parameters()),amp_bfloat16=use_amp))
    return best_path


def load_teacher(run,seed,device,smoke=False):
    path = Path(run)/f'seed_{seed}'/('smoke' if smoke else 'teacher')/'best.pt'
    ck = load_torch(path,device)
    model = Transformer(ck['model_config']).to(device).float().eval()
    model.load_state_dict(ck['model'])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
