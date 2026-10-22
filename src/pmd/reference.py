"""Known task references (§3/A3), never used to fit an activation encoder.

This diagonal instrument reproduces the oracle on supported post-clue histories.
It is not the learned Density2 model and is never used as its initialization.
"""
from __future__ import annotations

import math
import torch
from .generator import A,B,END,ONE,QUERY,RESET,ZERO,Stage,advance_stage,oracle_state


def oracle_density(history):
    belief=float(oracle_state(history).belief)
    return torch.diag(torch.tensor([1-belief,belief],dtype=torch.complex128))


def diagonal_branches(stage,device='cpu'):
    eye=torch.eye(2,dtype=torch.complex128,device=device)
    k=torch.zeros(8,2,2,2,dtype=torch.complex128,device=device)
    if stage==Stage.WAIT:
        for token,probability in ((A,.4),(B,.4),(QUERY,.1)):
            k[token,0]=math.sqrt(probability)*eye
        k[RESET,0]=math.sqrt(.05)*eye
        k[RESET,1]=math.sqrt(.05)*eye.flip(0)
    elif stage==Stage.ANSWER:
        k[ZERO,0,0,0]=1
        k[ONE,0,1,1]=1
    elif stage in (Stage.END_PENDING,Stage.STOP):
        k[END,0]=eye
    else:
        raise ValueError('The known generator has no reference law for an invalid prefix.')
    return k


def diagonal_predict(rho,stage):
    k=diagonal_branches(stage,rho.device)
    children=(k@rho@k.mH).sum(1)
    return children.diagonal(dim1=-2,dim2=-1).sum(-1).real


def diagonal_update(rho,stage,token):
    k=diagonal_branches(stage,rho.device)[token]
    child=(k@rho@k.mH).sum(0)
    probability=child.trace().real
    if probability<=0:
        raise ValueError('Cannot condition the oracle on an impossible observation.')
    return child/probability,advance_stage(stage,token)
