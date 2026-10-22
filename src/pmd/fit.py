"""Fit surrogates on clean histories only, with equal offsets and clean-state consistency."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch

from .generator import RESET,Stage,advance_stage
from .io import (append_jsonl,load_torch,progress,read_json,save_torch,set_seed,sha256,
                 synchronize,write_json)
from .metrics import kl
from .surrogates import make_surrogate
from .banks import unique_state_indexes


class TrainingData:
    def __init__(self,bank,source,prep,device):
        self.x = torch.as_tensor(prep.transform(bank[source]),device=device)
        self.p = torch.as_tensor(bank['teacher'],device=device)
        self.stage = torch.as_tensor(bank['stage'],device=device)
        self.pos = torch.as_tensor(bank['position'],device=device)
        self.active = torch.as_tensor(bank['active'],device=device)
        self.tokens = torch.as_tensor(bank['supplied'],device=device)
        observed_reset=np.concatenate((np.zeros((len(bank['supplied']),1),bool),
                                       np.cumsum(bank['supplied']==RESET,axis=1)>0),axis=1)
        observed_reset|=bank['reset'].astype(bool)[:,None]
        labels=bank['clue'][:,None]*2+observed_reset.astype(int)
        self.group = torch.as_tensor(labels,device=device)
        self.strata = []
        # Deduplicate complete prefixes, including continuation states. Labels only
        # balance the loss; they are never arguments to an encoder or recurrence.
        row,col = unique_state_indexes(bank)
        for clue in (0,1):
            for reset in (0,1):
                keep=labels[row,col]==clue*2+reset
                self.strata.append(np.column_stack((row[keep],col[keep])))
        self.n = len(bank['teacher'])

    def batch(self,rng,size):
        pairs = np.concatenate([ix[rng.integers(len(ix),size=size//4+(j<size%4))]
                                for j,ix in enumerate(self.strata) if len(ix)])
        rng.shuffle(pairs)
        # Continuation states are also fitting roots (§A5). Roots beyond offset 0
        # have shorter available rollouts; padding is never treated as evidence.
        return torch.as_tensor(pairs[:,0],device=self.x.device),torch.as_tensor(pairs[:,1],device=self.x.device)


def balanced_mean(values,valid,groups):
    sums=torch.zeros(4,device=values.device,dtype=values.dtype).scatter_add_(0,groups,values*valid)
    counts=torch.zeros_like(sums).scatter_add_(0,groups,valid.to(values.dtype))
    return (sums/counts.clamp_min(1))[counts>0].mean()


def rollout_loss(model,data,indexes,starts,consistency_weight=.1):
    z = model.encode(data.x[indexes,starts])
    loss = torch.zeros((),device=z.device)
    consistency = torch.zeros_like(loss)
    terms,consistency_terms = 0,0
    rows = indexes
    groups=data.group[rows,starts]
    for offset in range(5):
        cols = starts+offset
        in_bounds = cols<5
        safe = cols.clamp(max=4)
        valid = in_bounds & data.active[rows,safe]
        if not valid.any():
            break
        stage,pos = data.stage[rows,safe],data.pos[rows,safe]
        q = model.predict(z,stage,pos)
        loss = loss+balanced_mean(kl(data.p[rows,safe],q),valid,groups)
        terms += 1
        if offset:
            target = model.encode(data.x[rows,safe]).detach()
            consistency = consistency+balanced_mean(model.distance(z,target).square(),valid,groups)
            consistency_terms += 1
        if offset<4:
            tokens = data.tokens[rows,safe.clamp(max=3)]
            z = model.update(z,tokens,stage,pos)
    return loss/max(terms,1)+consistency_weight*consistency/max(consistency_terms,1)


@torch.no_grad()
def validation(model,data,batch=256,consistency_weight=.1):
    model.eval()
    scores=[]
    # Every distinct clean prefix, equal clue/reset group weighting.
    for pairs in data.strata:
        loss,count=0.,0
        for start in range(0,len(pairs),batch):
            current=torch.as_tensor(pairs[start:start+batch],device=data.x.device)
            loss+=float(rollout_loss(model,data,current[:,0],current[:,1],consistency_weight))*len(current)
            count+=len(current)
        if count:
            scores.append(loss/count)
    return float(np.mean(scores))


def fit_one(cfg,run,seed,source,family,restart,fit_data,val_data,device,time_budget=None,resume=False):
    directory = Path(run)/f'seed_{seed}'/'surrogates'/source/family/f'restart_{restart}'
    directory.mkdir(parents=True,exist_ok=True)
    final = directory/'fit_summary.json'
    if final.exists():
        return read_json(final)
    set_seed(seed*1000+restart)
    model = make_surrogate(family,cfg['analysis']['operator_epsilon']).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=cfg['analysis']['learning_rate'],
                                 betas=tuple(cfg['analysis']['betas']),weight_decay=0.)
    rng = np.random.default_rng(seed*1000+restart)
    step,best,elapsed_base = 0,float('inf'),0.
    if (directory/'last.pt').exists():
        if not resume:
            raise FileExistsError(f'Incomplete fit at {directory}; use --resume.')
        ck = load_torch(directory/'last.pt',device)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        step,best,elapsed_base = ck['step'],ck['best'],ck['training_seconds']
        rng.bit_generator.state = ck['rng']
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    measured_train = elapsed_base
    mandatory = cfg['analysis']['steps']
    # Classical fits get both >=3000 steps and the density restart's measured compute time.
    while step<mandatory or (time_budget is not None and measured_train<time_budget):
        synchronize(device)
        tick = time.perf_counter()
        model.train()
        ix,cols = fit_data.batch(rng,cfg['analysis']['batch_size'])
        optimizer.zero_grad(set_to_none=True)
        loss = rollout_loss(model,fit_data,ix,cols,cfg['analysis']['consistency_weight'])
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Non-finite surrogate loss at {source}/{family}/{restart}.')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['analysis']['clip'])
        optimizer.step()
        synchronize(device)
        measured_train += time.perf_counter()-tick
        step += 1
        append_jsonl(directory/'train.jsonl',dict(step=step,loss=float(loss.detach()),training_seconds=measured_train))
        finished = step>=mandatory and (time_budget is None or measured_train>=time_budget)
        if step%cfg['analysis']['validation_every']==0 or finished:
            score = validation(model,val_data,consistency_weight=cfg['analysis']['consistency_weight'])
            improve = score<best
            best = min(best,score)
            ck = dict(model=model.state_dict(),family=family,source=source,restart=restart,seed=seed,
                      step=step,best=best,optimizer=optimizer.state_dict(),rng=rng.bit_generator.state,
                      training_seconds=measured_train,epsilon=cfg['analysis']['operator_epsilon'])
            save_torch(directory/'last.pt',ck)
            if improve:
                save_torch(directory/'best.pt',ck)
            append_jsonl(directory/'validation.jsonl',dict(step=step,loss=score))
            progress(f'seed={seed} source={source} family={family} restart={restart} step={step} validation={score:.6f}')
    result = dict(family=family,source=source,restart=restart,seed=seed,steps=step,
                  best_validation=best,training_seconds=measured_train,
                  wall_seconds=elapsed_base+time.perf_counter()-start,time_budget_seconds=time_budget,
                  peak_vram_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                  parameter_counts=model.counts(),best_checkpoint=str(directory/'best.pt'))
    write_json(final,result)
    return result


def fit_all(cfg,run,seed,fit_bank,val_bank,device,resume=False):
    from .features import load_preprocessor
    for source in cfg['analysis']['sources']:
        prep = load_preprocessor(run,seed,source)
        fit_data = TrainingData(fit_bank,source,prep,device)
        val_data = TrainingData(val_bank,source,prep,device)
        density_times = {}
        families = sorted(cfg['analysis']['families'],key=lambda x:x!='density2')
        for family in families:
            outcomes = []
            for restart in cfg['analysis']['restarts']:
                budget = density_times.get(restart) if family!='density2' and cfg['analysis']['time_match_baselines'] else None
                result = fit_one(cfg,run,seed,source,family,restart,fit_data,val_data,device,budget,resume)
                outcomes.append(result)
                if family=='density2':
                    density_times[restart] = result['training_seconds']
            best = min(outcomes,key=lambda x:x['best_validation'])
            write_json(Path(run)/f'seed_{seed}'/'surrogates'/source/family/'selection.json',
                       dict(best=best,restarts=outcomes,selection_split='analysis_val',
                            checkpoint_sha256=sha256(Path(run)/f'seed_{seed}'/'surrogates'/source/family/f"restart_{best['restart']}"/'best.pt')))
        del fit_data,val_data
    lock_selection(cfg,run,seed)


def lock_selection(cfg,run,seed):
    root = Path(run)/f'seed_{seed}'
    paths = [root/'teacher'/'best.pt']
    paths += [root/'preprocessing'/f'{s}.pkl' for s in cfg['analysis']['sources']]
    paths += [root/'surrogates'/s/f/'selection.json' for s in cfg['analysis']['sources'] for f in cfg['analysis']['families']]
    for s in cfg['analysis']['sources']:
        for f in cfg['analysis']['families']:
            selection=read_json(root/'surrogates'/s/f/'selection.json')
            paths.append(root/'surrogates'/s/f/f"restart_{selection['best']['restart']}"/'best.pt')
    write_json(root/'selection_lock.json',dict(files={str(p.relative_to(root)):sha256(p) for p in paths},
               fitted_on_interventions=False,test_results_used_for_selection=False))


def load_surrogate(run,seed,source,family,device):
    selection = read_json(Path(run)/f'seed_{seed}'/'surrogates'/source/family/'selection.json')
    path = Path(run)/f'seed_{seed}'/'surrogates'/source/family/f"restart_{selection['best']['restart']}"/'best.pt'
    if not path.exists():
        path = Path(selection['best']['best_checkpoint'])
    if sha256(path)!=selection['checkpoint_sha256']:
        raise RuntimeError(f'Selected checkpoint hash mismatch: {path}')
    ck = load_torch(path,device)
    model = make_surrogate(family,ck['epsilon']).to(device).eval()
    model.load_state_dict(ck['model'])
    for p in model.parameters():
        p.requires_grad_(False)
    return model
