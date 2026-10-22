from __future__ import annotations

import numpy as np
import torch


def tv(p,q):
    if isinstance(p,torch.Tensor):
        return (p-q).abs().sum(-1)*.5
    return np.abs(np.asarray(p)-np.asarray(q)).sum(-1)*.5


def kl(p,q):
    # Zero target terms contribute zero; never discard impossible-token mass in q.
    return (p*(p.clamp_min(1e-30).log()-q.clamp_min(1e-30).log())).sum(-1)


def summary(values):
    x = np.asarray(values,dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {'n':0,'mean':None,'median':None,'p95':None,'min':None,'max':None}
    return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),
                p95=float(np.quantile(x,.95)),min=float(x.min()),max=float(x.max()))


def family_bootstrap(rows,value,family='family',samples=2000,seed=719,statistic='mean'):
    """Resample whole families, retaining every condition/position/arm in a family."""
    groups = {}
    for r in rows:
        if r.get(value) is not None and np.isfinite(r[value]):
            groups.setdefault(str(r[family]),[]).append(float(r[value]))
    if not groups:
        return {'estimate':None,'lower':None,'upper':None,'families':0}
    arrays = list(map(np.asarray,groups.values()))
    agg = np.mean if statistic=='mean' else np.median
    estimate = float(agg(np.concatenate(arrays)))
    rng = np.random.default_rng(seed)
    if statistic=='mean':
        # Exactly the same whole-family weighted mean, without repeatedly
        # concatenating potentially millions of condition rows in Python.
        sums=np.array([a.sum() for a in arrays]);counts=np.array([len(a) for a in arrays])
        boots=[]
        for start in range(0,samples,128):
            ix=rng.integers(len(arrays),size=(min(128,samples-start),len(arrays)))
            boots.extend((sums[ix].sum(1)/counts[ix].sum(1)).tolist())
    else:
        boots = [float(agg(np.concatenate([arrays[i] for i in rng.integers(len(arrays),size=len(arrays))]))) for _ in range(samples)]
    return dict(estimate=estimate,lower=float(np.quantile(boots,.025)),upper=float(np.quantile(boots,.975)),families=len(arrays))
