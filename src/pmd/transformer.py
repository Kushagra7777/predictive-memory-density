"""Small explicit-attention pre-LN transformer. Cache objects are branch-safe (§4.1)."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Cache:
    keys: list[torch.Tensor]
    values: list[torch.Tensor]
    valid: torch.Tensor

    def clone(self):
        return Cache([x.clone() for x in self.keys], [x.clone() for x in self.values], self.valid.clone())

    def detach(self):
        return Cache([x.detach() for x in self.keys], [x.detach() for x in self.values], self.valid.detach())

    def select(self, indexes):
        return Cache([x[indexes] for x in self.keys], [x[indexes] for x in self.values], self.valid[indexes])

    @property
    def positions(self):
        return self.valid.long().sum(-1)-1


@dataclass
class Forward:
    logits: torch.Tensor
    residuals: torch.Tensor  # [batch,new_position,layer,width], pre-final-LN
    cache: Cache


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads, self.head_width = heads, width//heads
        self.qkv = nn.Linear(width, 3*width, bias=True)
        self.output = nn.Linear(width, width, bias=True)

    def forward(self, x, prior_k, prior_v, valid):
        b, length, width = x.shape
        q, k, v = self.qkv(x).view(b,length,3,self.heads,self.head_width).unbind(2)
        q,k,v = [y.transpose(1,2) for y in (q,k,v)]
        past = 0 if prior_k is None else prior_k.shape[2]
        if prior_k is not None:
            k, v = torch.cat((prior_k,k),dim=2), torch.cat((prior_v,v),dim=2)
        scores = (q @ k.transpose(-2,-1)) / math.sqrt(self.head_width)
        causal = torch.arange(k.shape[2],device=x.device)[None,:] <= (past+torch.arange(length,device=x.device))[:,None]
        allowed = causal[None,None,:,:] & valid[:,None,None,:]
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        probs = scores.softmax(-1)
        result = (probs @ v).transpose(1,2).contiguous().view(b,length,width)
        return self.output(result), k, v


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg['width']
        self.ln1 = nn.LayerNorm(d)
        self.attn = Attention(d,cfg['heads'])
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d,cfg['ff_width']), nn.GELU(), nn.Linear(cfg['ff_width'],d))

    def forward(self,x,k,v,valid):
        a,k,v = self.attn(self.ln1(x),k,v,valid)
        x = x+a
        x = x+self.mlp(self.ln2(x))
        return x,k,v


class Transformer(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.config = dict(cfg)
        self.tokens = nn.Embedding(cfg['vocab_size'],cfg['width'])
        self.positions = nn.Embedding(cfg['context'],cfg['width'])
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg['layers'])])
        self.final_norm = nn.LayerNorm(cfg['width'])
        self.head = nn.Linear(cfg['width'],cfg['vocab_size'],bias=True)
        self.apply(self._init)

    def _init(self,module):
        if isinstance(module,(nn.Linear,nn.Embedding)):
            nn.init.normal_(module.weight,std=self.config.get('init_std',.02))
            if isinstance(module,nn.Linear):
                nn.init.zeros_(module.bias)
        elif isinstance(module,nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self,ids,mask=None,cache=None):
        if mask is None:
            mask = torch.ones_like(ids,dtype=torch.bool)
        prefix = 0 if cache is None else cache.valid.long().sum(-1,keepdim=True)
        pos = prefix + mask.long().cumsum(-1)-1
        pos = pos.clamp(min=0)
        if int(pos.max()) >= self.config['context']:
            raise ValueError('Position exceeds context; do not silently truncate the cache.')
        x = self.tokens(ids)+self.positions(pos)
        valid = mask if cache is None else torch.cat((cache.valid,mask),-1)
        residuals,keys,values = [],[],[]
        for i,block in enumerate(self.blocks):
            k = None if cache is None else cache.keys[i]
            v = None if cache is None else cache.values[i]
            x,k,v = block(x,k,v,valid)
            residuals.append(x)
            keys.append(k)
            values.append(v)
        return Forward(self.head(self.final_norm(x)),torch.stack(residuals,2),Cache(keys,values,valid))

    def optimizer_groups(self,weight_decay):
        decay,no_decay = [],[]
        for name,p in self.named_parameters():
            (decay if p.ndim >= 2 and not name.startswith(('tokens.','positions.')) else no_decay).append(p)
        return [{'params':decay,'weight_decay':weight_decay},{'params':no_decay,'weight_decay':0.0}]


def shifted_loss(output,ids,mask):
    loss = F.cross_entropy(output.logits[:,:-1].transpose(1,2),ids[:,1:],reduction='none')
    valid = mask[:,1:]
    return (loss*valid).sum()/valid.sum().clamp(min=1), int(valid.sum())


def cache_features(cache,current_slots=None):
    """Clue, current, and strictly intermediate mean K/V: 4*6*128=3072 reals."""
    b,n = cache.valid.shape
    device = cache.valid.device
    if current_slots is None:
        current_slots = (torch.arange(n,device=device)[None,:]*cache.valid).max(-1).values
    row = torch.arange(b,device=device)
    middle = cache.valid & (torch.arange(n,device=device)[None,:] > 17) & (torch.arange(n,device=device)[None,:] < current_slots[:,None])
    parts = []
    for k,v in zip(cache.keys,cache.values):
        # Convert to [batch,position,all_heads] in fixed head-major coordinate order.
        kk,vv = [z.transpose(1,2).flatten(2) for z in (k,v)]
        den = middle.sum(-1,keepdim=True).clamp(min=1)
        parts.extend([kk[:,17],vv[:,17],kk[row,current_slots],vv[row,current_slots],
                      (kk*middle[:,:,None]).sum(1)/den,(vv*middle[:,:,None]).sum(1)/den])
    return torch.cat(parts,-1)


def extract_at(output,slots):
    """Use full-forward tensors only up to each scored position, avoiding future leakage."""
    b = output.logits.shape[0]
    row = torch.arange(b,device=slots.device)
    res = output.residuals[row,slots]
    n = output.cache.valid.shape[1]
    prefix_valid = output.cache.valid & (torch.arange(n,device=slots.device)[None,:] <= slots[:,None])
    cache = Cache(output.cache.keys,output.cache.values,prefix_valid)
    return {'last_residual':res[:,-1], 'all_layers':res.flatten(1),
            'cache_summary':cache_features(cache,slots)}


def exchange(recipient,donor,earlier,current):
    if not torch.equal(recipient.valid,donor.valid):
        raise ValueError('Swap histories must have identical masks, lengths and positions.')
    out = recipient.clone()
    last = (torch.arange(out.valid.shape[1],device=out.valid.device)[None,:]*out.valid).max(-1).values
    positions = torch.arange(out.valid.shape[1],device=last.device)[None,:]
    swap = ((positions < last[:,None]) & earlier) | ((positions == last[:,None]) & current)
    swap &= out.valid
    for i in range(len(out.keys)):
        for target,source in [(out.keys[i],donor.keys[i]),(out.values[i],donor.values[i])]:
            target.copy_(torch.where(swap[:,None,:,None],source,target))
    return out
