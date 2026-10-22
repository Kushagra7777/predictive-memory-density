from fractions import Fraction
import numpy as np
import pytest
import torch

from pmd.config import load_config
from pmd.generator import *
from pmd.metrics import family_bootstrap
from pmd.surrogates import make_surrogate,rho_from_bloch,inverse_sqrt_2x2
from pmd.transformer import Transformer,cache_features,exchange,extract_at,shifted_loss


torch.set_num_threads(2)


@pytest.fixture
def cfg():
    return load_config('configs/tiny.yaml')


def test_exact_probabilities_and_reset():
    assert verify_generator()['horizon_tv_exact']==['0','3/50','27/250']
    h=make_root(12,0,[A,B])
    assert oracle_distribution(h+[QUERY])[ONE]==Fraction(1,5)
    assert oracle_distribution(h+[RESET,QUERY])[ONE]==Fraction(1,2)


def test_group_splits_disjoint_and_cover_every_nuisance():
    pools=[set(family_pool(s).tolist()) for s in SPLITS]
    assert len(set.union(*pools))==65536
    for i,pool in enumerate(pools):
        assert all(not pool.intersection(other) for other in pools[i+1:])


def test_generation_and_censoring():
    rng=np.random.default_rng(19)
    for _ in range(200):
        h=episode(rng,max_context=24)
        assert len(h)<=24
        assert h[0]==BOS and set(h[1:17])<={A,B}
        assert h[17] in (ZERO,ONE)
        assert len(h)==24 or h[-1]==END
        if END in h:
            assert h[-1]==END
    assert advance_stage(Stage.INVALID,END)==Stage.STOP


def test_parameter_counts(cfg):
    assert sum(p.numel() for p in Transformer(cfg['model']).parameters())==803592
    expected={'density2':(4227,1024),'classical4':(4227,1024),'vector3':(4227,1010),'classical2':(4161,256)}
    for name,(encoder,dynamics) in expected.items():
        counts=make_surrogate(name).counts()
        assert counts['encoder']==encoder
        assert counts['dynamics_readout']==dynamics


@pytest.mark.parametrize('family',['density2','classical4','vector3','classical2'])
def test_surrogate_probabilities_gradients_and_absorption(family):
    torch.manual_seed(3)
    model=make_surrogate(family)
    z=model.encode(torch.randn(8,128))
    stage=torch.tensor([0,1,2,3,0,1,2,4]);pos=torch.arange(18,26)
    p=model.predict(z,stage,pos)
    assert (p>=-1e-7).all()
    assert torch.allclose(p.sum(-1),torch.ones(8),atol=2e-6)
    assert p[-1,END]==1 and p[-1].sum()==1
    nxt=model.update(z,torch.arange(8),stage,pos)
    loss=-p[:7].clamp_min(1e-9).log().mean()+nxt.square().mean()
    loss.backward()
    assert all(t.grad is None or torch.isfinite(t.grad).all() for t in model.parameters())
    if family=='density2':
        eigen=torch.linalg.eigvalsh(rho_from_bloch(nxt))
        assert eigen.min()>-1e-6


def test_density_completeness_and_inverse_sqrt():
    model=make_surrogate('density2')
    k=model.operators(torch.tensor([0,1,2,3]),torch.tensor([18,29,40,62]))
    eye=(k.mH@k).sum((1,2))
    assert torch.allclose(eye,torch.eye(2,dtype=torch.complex128).expand_as(eye),atol=1e-12)
    g=torch.eye(2,dtype=torch.complex128).repeat(2,1,1).requires_grad_()
    inverse_sqrt_2x2(g).real.sum().backward()
    assert torch.isfinite(g.grad).all()


def test_causal_mask_and_padding(cfg):
    torch.manual_seed(2)
    m=Transformer(cfg['model']).eval()
    h=make_root(9,0,[A,B])
    ids,mask=pad_histories([h],64)
    ids=torch.tensor(ids);mask=torch.tensor(mask)
    first=m(ids,mask)
    changed=ids.clone();changed[:,len(h):]=7
    second=m(changed,mask)
    assert torch.equal(first.logits[:,:len(h)],second.logits[:,:len(h)])
    changed=ids.clone();changed[:,len(h)-1]=7
    third=m(changed,mask)
    assert torch.allclose(first.logits[:,:len(h)-1],third.logits[:,:len(h)-1],atol=1e-7)
    loss,n=shifted_loss(first,ids,mask)
    assert n==len(h)-1 and torch.isfinite(loss)


def test_cached_forward_swaps_and_branch_cloning(cfg):
    from pmd.evaluate import prefill,teacher_step
    torch.manual_seed(9)
    m=Transformer(cfg['model']).eval()
    h0,h1=[make_root(7,c,[A,B,A,B]) for c in (0,1)]
    a,b=[prefill(m,[h],torch.device('cpu')) for h in (h0,h1)]
    baseline=teacher_step(m,a,torch.tensor([QUERY]))
    full=m(torch.tensor([h0+[QUERY]])).logits[:,-1]
    assert torch.allclose(baseline.probs,full.softmax(-1),atol=1e-6)
    original=a.cache.keys[0].clone()
    transplanted=exchange(a.cache,b.cache,True,True)
    assert torch.equal(a.cache.keys[0],original)
    assert transplanted.keys[0].data_ptr()!=a.cache.keys[0].data_ptr()
    assert torch.equal(transplanted.keys[0],b.cache.keys[0])
    partial=exchange(a.cache,b.cache,True,False)
    assert torch.equal(partial.keys[0][:,:,-1],a.cache.keys[0][:,:,-1])
    assert torch.equal(partial.keys[0][:,:,:-1],b.cache.keys[0][:,:,:-1])
    replay=m(torch.tensor([[QUERY]]),cache=transplanted).logits[:,-1]
    expected=m(torch.tensor([[QUERY]]),cache=b.cache).logits[:,-1]
    assert torch.equal(replay,expected)


def test_future_feature_extraction_does_not_leak(cfg):
    torch.manual_seed(8)
    m=Transformer(cfg['model']).eval()
    h=make_root(17,1,[A,B,A,B])
    a=m(torch.tensor([h+[QUERY,ONE,END]]))
    b=m(torch.tensor([h]))
    fa=extract_at(a,torch.tensor([len(h)-1]))
    fb=extract_at(b,torch.tensor([len(h)-1]))
    assert set(fa)=={'last_residual','all_layers','cache_summary'}
    for key in fa:
        assert torch.allclose(fa[key],fb[key],atol=1e-6)


def test_backward_through_cache(cfg):
    torch.manual_seed(4)
    m=Transformer(cfg['model'])
    h=make_root(19,0,[A,B])
    first=m(torch.tensor([h]))
    first.cache.values[0].retain_grad()
    second=m(torch.tensor([[QUERY]]),cache=first.cache)
    second.logits[0,0,ONE].backward()
    assert first.cache.values[0].grad is not None
    assert first.cache.values[0].grad.abs().sum()>0


def test_three_token_tree_mass_and_stopping(cfg):
    from pmd.evaluate import prefill,teacher_tree,surrogate_tree
    m=Transformer(cfg['model']).eval()
    state=prefill(m,[make_root(4,0,[A,B])],torch.device('cpu'))
    mass=teacher_tree(m,state,3)
    assert mass.shape==(1,512) and abs(float(mass.sum())-1)<1e-6
    # Every branch that emits '$' has only '$' afterwards, without renormalization.
    cube=mass.reshape(8,8,8)
    assert cube[END,:END].sum()==0 and cube[END,END,:END].sum()==0
    d=make_surrogate('density2')
    z=d.encode(torch.randn(1,128))
    q=surrogate_tree(d,z,state.stages,state.positions,3)
    assert abs(float(q.sum())-1)<1e-6


def test_bootstrap_groups_whole_families():
    rows=[dict(family=1,value=1.),dict(family=1,value=1.),dict(family=2,value=0.)]
    result=family_bootstrap(rows,'value',samples=30)
    assert result['families']==2 and result['lower']<=result['estimate']<=result['upper']
