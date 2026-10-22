"""Disjoint clean-history feature banks. No intervention outcomes enter fitting."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch

from .generator import (A,B,END,QUERY,Stage,TOKENS,advance_stage,family_pool,
                        make_root,oracle_distribution,oracle_state,pad_histories,sample_from_prefix)
from .io import progress,read_json,write_json
from .transformer import extract_at

SOURCES = ('last_residual','all_layers','cache_summary')


def root_records(count,split,seed,delays):
    rng = np.random.default_rng(seed)
    seen,rows = set(),[]
    attempts = 0
    while len(rows)<count:
        i = len(rows)
        clue,reset = i%2,(i//2)%2
        delay = int(delays[(i//4)%len(delays)])
        family = int(rng.choice(family_pool(split)))
        waiting = rng.choice([A,B],size=delay).tolist()
        h = make_root(family,clue,waiting,reset)
        attempts += 1
        if tuple(h) in seen:
            if attempts>count*200:
                raise RuntimeError('Could not generate enough distinct analysis histories.')
            continue
        seen.add(tuple(h))
        rows.append(dict(history=h,family=family,clue=clue,reset=reset,base_delay=delay))
    rng.shuffle(rows)
    return rows


def continuation(rng,root,steps,forced):
    if not forced:
        return sample_from_prefix(rng,root,steps)
    # Half use a supported query at a uniformly chosen offset 1..4 (§4.3).
    before = int(rng.integers(steps))
    out = rng.choice([A,B],size=before).tolist()+[QUERY]
    return out+sample_from_prefix(rng,root+out,steps-len(out))


@torch.no_grad()
def collect_chunk(model,records,rng,device,steps=4):
    histories = []
    continuations = []
    for i,row in enumerate(records):
        cont = continuation(rng,row['history'],steps,row.get('forced_query',bool(i%2)))
        continuations.append(cont)
        histories.append(row['history']+cont)
    tokens,mask = pad_histories(histories,model.config['context'])
    ids,valid = torch.as_tensor(tokens,device=device),torch.as_tensor(mask,device=device)
    output = model(ids,valid)
    b = len(records)
    features = {s:np.zeros((b,steps+1,n),np.float32) for s,n in [('last_residual',128),('all_layers',512),('cache_summary',3072)]}
    teacher = np.zeros((b,steps+1,8),np.float32)
    stage = np.full((b,steps+1),int(Stage.STOP),np.int64)
    position = np.zeros((b,steps+1),np.int64)
    belief = np.zeros((b,steps+1),np.float32)
    active = np.zeros((b,steps+1),bool)
    supplied = np.full((b,steps),END,np.int64)
    lengths = np.array([len(r['history']) for r in records])
    for i,cont in enumerate(continuations):
        supplied[i,:len(cont)] = cont
    for k in range(steps+1):
        slots = np.minimum(lengths-1+k,np.array([len(h)-1 for h in histories]))
        feat = extract_at(output,torch.as_tensor(slots,device=device))
        p = output.logits[torch.arange(b,device=device),torch.as_tensor(slots,device=device)].float().softmax(-1).cpu().numpy()
        for i,row in enumerate(records):
            if k > len(continuations[i]):
                continue
            h = row['history']+continuations[i][:k]
            state = oracle_state(h)
            # Stop at termination. The absorbing endpoint is evaluated separately in trees.
            if state.stage==Stage.STOP:
                continue
            active[i,k] = True
            stage[i,k],position[i,k],belief[i,k] = state.stage,len(h)-1,float(state.belief)
            teacher[i,k] = p[i]
        for s in SOURCES:
            features[s][:,k] = feat[s].float().cpu().numpy()
    roots,_ = pad_histories([r['history'] for r in records],64)
    return dict(**features,teacher=teacher,stage=stage,position=position,belief=belief,active=active,
                supplied=supplied,roots=roots,root_length=lengths,
                family=np.array([r['family'] for r in records]),clue=np.array([r['clue'] for r in records]),
                reset=np.array([r['reset'] for r in records]),base_delay=np.array([r['base_delay'] for r in records]))


def collect_bank(cfg,run,seed,model,device,split,force=False):
    directory = Path(run)/f'seed_{seed}'/'banks'/split
    if (directory/'manifest.json').exists() and not force:
        return directory
    directory.mkdir(parents=True,exist_ok=True)
    tick=time.perf_counter()
    count = cfg['analysis']['roots_fit' if split=='fit' else 'roots_val']
    records = root_records(count,split,seed+(30000 if split=='fit' else 40000),cfg['analysis']['root_delays'])
    rng = np.random.default_rng(seed+(31000 if split=='fit' else 41000))
    forced_order=rng.permutation(count)
    for i,row in enumerate(records):
        row['forced_query']=bool(forced_order[i]<count//2)
    chunk_size = cfg['analysis']['collection_batch']
    shard_size = cfg['analysis']['shard_size']
    parts,rows,names = [],0,[]
    for start in range(0,count,chunk_size):
        chunk = collect_chunk(model,records[start:start+chunk_size],rng,device,cfg['analysis']['rollout'])
        parts.append(chunk)
        rows += len(chunk['teacher'])
        if rows>=shard_size or start+chunk_size>=count:
            joined = {k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
            name = f'shard_{len(names):05d}.npz'
            np.savez_compressed(directory/name,**joined)
            names.append(name)
            parts,rows = [],0
            progress(f'seed={seed} collect={split} roots={min(start+chunk_size,count)}/{count}')
    write_json(directory/'manifest.json',dict(split=split,root_count=count,shards=names,
               frozen_teacher=True,patched_caches_used=False,
               feature_sizes={'last_residual':128,'all_layers':512,'cache_summary':3072},
               reset_delay_convention='base_delay counts fillers; reset roots append R, adding one actual waiting token',
               fitting_half_forced_queries=True,forced_query_roots=count//2,collection_seconds=time.perf_counter()-tick,
               feature_bank_bytes=sum((directory/name).stat().st_size for name in names)))
    return directory


def load_bank(directory):
    directory = Path(directory)
    manifest = read_json(directory/'manifest.json')
    values = {}
    for name in manifest['shards']:
        with np.load(directory/name) as shard:
            for key in shard.files:
                values.setdefault(key,[]).append(shard[key])
    return {key:np.concatenate(parts) for key,parts in values.items()}


def unique_state_indexes(bank):
    seen,indexes = set(),[]
    for i,k in zip(*np.where(bank['active'])):
        h = tuple(bank['roots'][i,:bank['root_length'][i]])+tuple(bank['supplied'][i,:k])
        if h not in seen:
            seen.add(h)
            indexes.append((i,k))
    return tuple(np.asarray(indexes,dtype=np.int64).T)
