"""Exactly budgeted density, classical-HMM and recurrent surrogates (§4.3-4.4/A2-A4)."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from .generator import END, Stage


def rho_from_bloch(r):
    x,y,z = r.unbind(-1)
    rho = torch.zeros((*r.shape[:-1],2,2),device=r.device,dtype=torch.complex128)
    rho[...,0,0] = (1+z.double())/2
    rho[...,1,1] = (1-z.double())/2
    rho[...,0,1] = torch.complex(x.double(),-y.double())/2
    rho[...,1,0] = torch.complex(x.double(),y.double())/2
    return rho


def bloch_from_rho(rho):
    return torch.stack((2*rho[...,0,1].real,-2*rho[...,0,1].imag,
                        rho[...,0,0].real-rho[...,1,1].real),-1).float()


def inverse_sqrt_2x2(g):
    """Analytic positive-definite inverse sqrt; no eigenvector derivatives (§A3)."""
    a = torch.linalg.det(g).real.sqrt()
    c = (g.diagonal(dim1=-2,dim2=-1).sum(-1).real+2*a).sqrt()
    eye = torch.eye(2,device=g.device,dtype=g.dtype)
    return c[...,None,None]*torch.linalg.inv(g+a[...,None,None]*eye)


class Surrogate(nn.Module):
    family = ''

    def __init__(self,outputs=3):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(128,32),nn.GELU(),nn.Linear(32,outputs))

    def encode(self,features):
        raise NotImplementedError

    @staticmethod
    def stopped(q,stages):
        terminal = F.one_hot(torch.full_like(stages,END),8).to(q.dtype)
        return torch.where((stages==Stage.STOP)[:,None],terminal,q)

    def counts(self):
        encoder = sum(p.numel() for p in self.encoder.parameters())
        total = sum(p.numel() for p in self.parameters())
        return dict(encoder=encoder,dynamics_readout=total-encoder,total=total,
                    stored_real_coordinates=1 if self.family=='classical2' else 3,
                    side_state='stage and position, two integers')

    def predict(self,z,stages,positions):
        raise NotImplementedError

    def update(self,z,tokens,stages,positions):
        raise NotImplementedError


class Density2(Surrogate):
    family = 'density2'

    def __init__(self,epsilon=1e-6):
        super().__init__(3)
        self.raw = nn.Parameter(torch.randn(4,2,8,2,2,2,2)*.15)
        self.epsilon = epsilon

    def encode(self,features):
        u = self.encoder(features)
        return u/torch.sqrt(1+(u*u).sum(-1,keepdim=True))

    def operators(self,stages,positions):
        a = self.raw[stages.clamp(max=3)]
        a = a[:,0] + (positions.float()/64)[:,None,None,None,None,None]*a[:,1]
        b = torch.complex(a[...,0].double(),a[...,1].double())
        eye = torch.eye(2,device=b.device,dtype=b.dtype)
        regularizer = (math.sqrt(self.epsilon/8)*eye).expand(b.shape[0],8,1,2,2)
        b = torch.cat((b,regularizer),2)
        g = (b.mH @ b).sum((1,2))
        return b @ inverse_sqrt_2x2(g)[:,None,None,:,:]

    def instrument(self,z,stages,positions):
        k = self.operators(stages,positions)
        rho = rho_from_bloch(z)
        children = (k @ rho[:,None,None,:,:] @ k.mH).sum(2)
        q = children.diagonal(dim1=-2,dim2=-1).sum(-1).real
        return children,q

    def predict(self,z,stages,positions):
        return self.stopped(self.instrument(z,stages,positions)[1].float(),stages)

    def update(self,z,tokens,stages,positions):
        children,q = self.instrument(z,stages,positions)
        rows = torch.arange(len(z),device=z.device)
        selected = children[rows,tokens]/q[rows,tokens,None,None]
        updated = bloch_from_rho(selected)
        return torch.where((stages==Stage.STOP)[:,None],z,updated)

    def distance(self,z,other):
        return torch.linalg.vector_norm(z-other,dim=-1)/2


class Classical(Surrogate):
    def __init__(self,states):
        super().__init__(states-1)
        self.states = states
        self.family = f'classical{states}'
        self.raw = nn.Parameter(torch.randn(4,2,8,states,states)*.15)

    def encode(self,features):
        u = self.encoder(features)
        return F.softmax(torch.cat((u,torch.zeros_like(u[:,:1])),-1),-1)[:,:-1]

    @staticmethod
    def full(z):
        return torch.cat((z,1-z.sum(-1,keepdim=True)),-1)

    def instrument(self,z,stages,positions):
        scores = self.raw[stages.clamp(max=3)]
        scores = scores[:,0]+(positions.float()/64)[:,None,None,None]*scores[:,1]
        b,d = len(z),self.states
        matrix = scores.permute(0,3,1,2).reshape(b,d,8*d).softmax(-1)
        matrix = matrix.view(b,d,8,d).permute(0,2,3,1)
        children = (matrix @ self.full(z)[:,None,:,None]).squeeze(-1)
        return children,children.sum(-1)

    def predict(self,z,stages,positions):
        return self.stopped(self.instrument(z,stages,positions)[1],stages)

    def update(self,z,tokens,stages,positions):
        children,q = self.instrument(z,stages,positions)
        rows = torch.arange(len(z),device=z.device)
        new = (children[rows,tokens]/q[rows,tokens,None])[:,:-1]
        return torch.where((stages==Stage.STOP)[:,None],z,new)

    def distance(self,z,other):
        return (self.full(z)-self.full(other)).abs().sum(-1)/2


class Vector3(Surrogate):
    family = 'vector3'

    def __init__(self):
        super().__init__(3)
        self.transition = nn.Sequential(nn.Linear(16,27),nn.GELU(),nn.Linear(27,3))
        self.readout = nn.Sequential(nn.Linear(8,27),nn.GELU(),nn.Linear(27,8))

    def encode(self,features):
        return torch.tanh(self.encoder(features))

    @staticmethod
    def context(stages,positions):
        return torch.cat((F.one_hot(stages.clamp(max=3),4).float(),positions.float()[:,None]/64),-1)

    def predict(self,z,stages,positions):
        q = self.readout(torch.cat((z,self.context(stages,positions)),-1)).softmax(-1)
        return self.stopped(q,stages)

    def update(self,z,tokens,stages,positions):
        x = torch.cat((z,F.one_hot(tokens,8).float(),self.context(stages,positions)),-1)
        new = torch.tanh(self.transition(x))
        return torch.where((stages==Stage.STOP)[:,None],z,new)

    def distance(self,z,other):
        return torch.linalg.vector_norm(z-other,dim=-1)/(2*math.sqrt(3))


def make_surrogate(family,epsilon=1e-6):
    if family == 'density2':
        return Density2(epsilon)
    if family == 'vector3':
        return Vector3()
    if family in ('classical4','classical2'):
        return Classical(int(family[-1]))
    raise ValueError(f'Unknown surrogate: {family}')
