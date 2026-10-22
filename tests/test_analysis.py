from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import torch

from pmd.banks import root_records,collect_chunk,unique_state_indexes
from pmd.config import load_config
from pmd.features import Preprocessor
from pmd.fit import TrainingData,balanced_mean,rollout_loss,lock_selection
from pmd.generator import *
from pmd.io import read_json,write_json
from pmd.reference import diagonal_predict,diagonal_update,oracle_density
from pmd.surrogates import make_surrogate
from pmd.transformer import Transformer


def test_reference_density_agrees_with_exact_oracle_and_reset():
    for clue in (0,1):
        h=make_root(12,clue,[A,B,A,B])
        rho=oracle_density(h)
        expected=np.array(list(map(float,oracle_distribution(h))))
        assert np.allclose(diagonal_predict(rho,Stage.WAIT),expected,atol=1e-15)
        reset,s=diagonal_update(rho,Stage.WAIT,RESET)
        assert torch.allclose(reset,torch.eye(2,dtype=torch.complex128)/2,atol=1e-15)
        next_rho,s=diagonal_update(rho,Stage.WAIT,QUERY)
        assert abs(float(diagonal_predict(next_rho,s)[ONE])-(.8 if clue else .2))<1e-15
        revealed,s=diagonal_update(next_rho,s,ONE)
        assert diagonal_predict(revealed,s)[END]==1


def test_pca_is_frozen_unwhitened_and_uses_only_fit_data():
    rng=np.random.default_rng(7)
    fit=rng.normal(size=(160,512)).astype(np.float32)
    prep=Preprocessor().fit(fit)
    original=prep.mean.copy()
    validation=rng.normal(100,10,size=(30,512)).astype(np.float32)
    result=prep.transform(validation)
    assert result.shape==(30,128) and np.array_equal(prep.mean,original)
    manual=(((fit-prep.mean)/prep.scale)-prep.pca_mean)@prep.components.T
    assert np.allclose(prep.transform(fit),manual,atol=2e-6)
    assert not np.allclose(prep.transform(fit).std(0),1,atol=.1)
    assert np.all(prep.scale>=1e-6)


def test_clean_bank_deduplication_and_no_diagnostic_inputs():
    cfg=load_config('configs/tiny.yaml')
    model=Transformer(cfg['model']).eval()
    records=root_records(16,'fit',71,[0,1])
    bank=collect_chunk(model,records,np.random.default_rng(2),'cpu')
    # Deliberately repeat every collected row. It must not double fitting prefixes.
    doubled={k:np.concatenate([v,v]) for k,v in bank.items()}
    assert len(unique_state_indexes(bank)[0])==len(unique_state_indexes(doubled)[0])
    prep=Preprocessor().fit(bank['last_residual'][:,0])
    data=TrainingData(doubled,'last_residual',prep,'cpu')
    assert sum(len(s) for s in data.strata)==len(unique_state_indexes(bank)[0])
    rows,starts=data.batch(np.random.default_rng(1),16)
    assert torch.bincount(data.group[rows,starts],minlength=4).tolist()==[4,4,4,4]
    assert data.x.shape[-1]==128
    loss=rollout_loss(make_surrogate('density2'),data,rows,starts)
    loss.backward()
    assert torch.isfinite(loss)


def test_balanced_mean_does_not_weight_longer_strata_more():
    values=torch.tensor([1.,1.,1.,0.]);valid=torch.ones(4,dtype=torch.bool)
    assert balanced_mean(values,valid,torch.tensor([0,0,0,1]))==.5


def test_loss_keeps_autonomous_state_and_averages_consistency_separately():
    from types import SimpleNamespace
    class Counter:
        def encode(self,x): return x[:,:1]*0
        def predict(self,z,s,p): return torch.full((len(z),8),1/8)
        def update(self,z,x,s,p): return z+1
        def distance(self,z,target): return (z-target).abs().squeeze(-1)
    data=SimpleNamespace(x=torch.zeros(4,5,128),p=torch.full((4,5,8),1/8),
                         active=torch.ones(4,5,dtype=torch.bool),stage=torch.zeros(4,5,dtype=torch.long),
                         pos=torch.zeros(4,5,dtype=torch.long),tokens=torch.zeros(4,4,dtype=torch.long),
                         group=torch.arange(4)[:,None].expand(4,5))
    loss=rollout_loss(Counter(),data,torch.arange(4),torch.zeros(4,dtype=torch.long))
    # Zero KL; autonomous distances 1,2,3,4; mean squared distance is 7.5.
    # Refreshing the state every step or dividing the second term by five fails.
    assert torch.isclose(loss,torch.tensor(.75))


def test_instrument_future_distance_bound_and_conditional_exception():
    from pmd.evaluate import surrogate_tree
    torch.manual_seed(9)
    model=make_surrogate('density2')
    a=model.encode(torch.randn(1,128));b=model.encode(torch.randn(1,128))
    stage=torch.tensor([0]);position=torch.tensor([25])
    pa=surrogate_tree(model,a,stage,position,3)
    pb=surrogate_tree(model,b,stage,position,3)
    assert ((pa-pb).abs().sum()/2)<=model.distance(a,b)+1e-6
    # The constructed task itself demonstrates amplification by conditioning.
    h0,h1=[make_root(9,c,[A,B,A,B]) for c in (0,1)]
    assert verify_generator()['horizon_tv_exact'][1]=='3/50'
    p0=np.array(list(map(float,oracle_distribution(h0+[QUERY]))))
    p1=np.array(list(map(float,oracle_distribution(h1+[QUERY]))))
    assert np.isclose(np.abs(p0-p1).sum()/2,.6)


def test_json_missing_values_are_valid_json(tmp_path):
    path=tmp_path/'stats.json'
    write_json(path,dict(x=np.float64(np.nan),y=np.array([np.inf,1.])))
    assert read_json(path)==dict(x=None,y=[None,1.])


def test_frozen_selection_detects_modified_checkpoint(tmp_path):
    from pmd.evaluate import verify_selection
    cfg=load_config('configs/tiny.yaml')
    cfg['analysis']['sources']=['last_residual'];cfg['analysis']['families']=['density2']
    root=tmp_path/'seed_17'
    for path in [root/'teacher'/'best.pt',root/'preprocessing'/'last_residual.pkl',
                 root/'surrogates'/'last_residual'/'density2'/'restart_101'/'best.pt']:
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'test artifact')
    write_json(root/'surrogates'/'last_residual'/'density2'/'selection.json',dict(best=dict(restart=101)))
    lock_selection(cfg,tmp_path,17)
    verify_selection(tmp_path,17)
    (root/'surrogates'/'last_residual'/'density2'/'restart_101'/'best.pt').write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='changed after sealing'):
        verify_selection(tmp_path,17)


def test_relocated_run_loads_its_own_checkpoint(tmp_path):
    from pmd.fit import load_surrogate
    from pmd.io import save_torch,sha256
    root=tmp_path/'new'/'seed_17'/'surrogates'/'cache_summary'/'density2'
    chosen=root/'restart_101'/'best.pt'
    old=tmp_path/'old'/'best.pt'
    model=make_surrogate('density2')
    save_torch(chosen,dict(model=model.state_dict(),epsilon=1e-6))
    save_torch(old,dict(model=make_surrogate('density2').state_dict(),epsilon=1e-6))
    write_json(root/'selection.json',dict(best=dict(restart=101,best_checkpoint=str(old)),checkpoint_sha256=sha256(chosen)))
    loaded=load_surrogate(tmp_path/'new',17,'cache_summary','density2','cpu')
    assert torch.equal(loaded.raw,model.raw)


def test_revised_settings_cannot_claim_the_frozen_formal_protocol(tmp_path):
    from pmd.cli import initialize
    cfg=load_config('configs/pilot.yaml')
    cfg['analysis']['steps']=3001
    with pytest.raises(ValueError,match='Formal profile differs'):
        initialize(cfg,tmp_path/'changed',torch.device('cpu'))


def test_time_matched_fit_reaches_budget_and_reuses_completed_snapshot(tmp_path):
    from pmd.fit import fit_one
    from pmd.io import sha256
    cfg=load_config('configs/tiny.yaml')
    cfg['analysis']['validation_every']=50
    bank=collect_chunk(Transformer(cfg['model']).eval(),root_records(16,'fit',3,[0,1]),np.random.default_rng(7),'cpu')
    prep=Preprocessor().fit(bank['last_residual'][:,0])
    data=TrainingData(bank,'last_residual',prep,'cpu')
    result=fit_one(cfg,tmp_path,17,'last_residual','classical4',101,data,data,torch.device('cpu'),time_budget=.05)
    assert result['steps']>=2 and result['training_seconds']>=.05
    last=tmp_path/'seed_17'/'surrogates'/'last_residual'/'classical4'/'restart_101'/'last.pt'
    before=sha256(last)
    again=fit_one(cfg,tmp_path,17,'last_residual','classical4',101,data,data,torch.device('cpu'),time_budget=.05,resume=True)
    assert result==again and sha256(last)==before
