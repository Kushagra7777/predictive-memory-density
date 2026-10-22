"""Held-out full-distribution, autonomous-update, and two-location cache tests (§5)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .generator import (A,B,END,ONE,QUERY,RESET,Stage,TOKENS,advance_stage,episode,family_pool,
                        make_root,oracle_distribution,oracle_state,pad_histories,sample_from_prefix)
from .io import progress,read_json,save_torch,sha256,synchronize,write_csv,write_json
from .metrics import summary,tv
from .transformer import Cache,cache_features,exchange,extract_at


def stage_tensor(stages,tokens):
    old = stages.detach().cpu().tolist()
    xs = tokens.detach().cpu().tolist()
    return torch.tensor([advance_stage(s,x) for s,x in zip(old,xs)],device=stages.device)


def absorb(probs,stages):
    end = F.one_hot(torch.full_like(stages,END),8).to(probs.dtype)
    return torch.where((stages==Stage.STOP)[:,None],end,probs)


def impossible_mass(probs,stages):
    allowed = torch.zeros_like(probs,dtype=torch.bool)
    for s,tokens in [(Stage.WAIT,[A,B,QUERY,RESET]),(Stage.ANSWER,[1,2]),
                     (Stage.END_PENDING,[END]),(Stage.STOP,[END])]:
        rows = stages==s
        for token in tokens:
            allowed[rows,token] = True
    allowed[stages==Stage.INVALID] = True
    return (probs*~allowed).sum(-1)


@dataclass
class TeacherState:
    cache: Cache
    probs: torch.Tensor
    features: dict
    stages: torch.Tensor
    positions: torch.Tensor

    def select(self,ix):
        return TeacherState(self.cache.select(ix),self.probs[ix],{k:v[ix] for k,v in self.features.items()},
                            self.stages[ix],self.positions[ix])


@torch.no_grad()
def prefill(model,histories,device):
    ids,valid = pad_histories(histories)
    ids,valid = torch.as_tensor(ids,device=device),torch.as_tensor(valid,device=device)
    output = model(ids,valid)
    slots = valid.long().sum(-1)-1
    stages = torch.tensor([oracle_state(h).stage for h in histories],device=device)
    probs = output.logits[torch.arange(len(histories),device=device),slots].softmax(-1)
    return TeacherState(output.cache.detach(),absorb(probs,stages),extract_at(output,slots),stages,slots)


@torch.no_grad()
def teacher_step(model,state,tokens):
    # Terminal paths are calculated only to keep batch shapes uniform. Their raw
    # logits are discarded: scored probabilities are fixed to absorbing '$'.
    output = model(tokens[:,None],cache=state.cache)
    stages = stage_tensor(state.stages,tokens)
    features = {'last_residual':output.residuals[:,-1,-1],
                'all_layers':output.residuals[:,-1].flatten(1),
                'cache_summary':cache_features(output.cache)}
    return TeacherState(output.cache,absorb(output.logits[:,-1].softmax(-1),stages),features,stages,state.positions+1)


@torch.no_grad()
def teacher_tree(model,state,horizon):
    b = len(state.probs)
    mass = torch.ones(b,1,device=state.probs.device)
    current = state
    for depth in range(horizon):
        mass = (mass.reshape(-1,1)*current.probs).reshape(b,-1)
        if depth+1<horizon:
            ix = torch.arange(len(current.probs),device=mass.device).repeat_interleave(8)
            tokens = torch.arange(8,device=mass.device).repeat(len(current.probs))
            current = teacher_step(model,current.select(ix),tokens)
    return mass


@torch.no_grad()
def surrogate_tree(model,z,stages,positions,horizon):
    b = len(z)
    mass = torch.ones(b,1,device=z.device)
    for depth in range(horizon):
        p = model.predict(z,stages,positions)
        mass = (mass.reshape(-1,1)*p).reshape(b,-1)
        if depth+1<horizon:
            count = len(z)
            z,stages,positions = z.repeat_interleave(8,0),stages.repeat_interleave(8),positions.repeat_interleave(8)
            tokens = torch.arange(8,device=z.device).repeat(count)
            z = model.update(z,tokens,stages,positions)
            stages,positions = stage_tensor(stages,tokens),positions+1
    return mass


def test_plan(cfg,run):
    path = Path(run)/'final_test_plan.json'
    if path.exists():
        return read_json(path)
    rng = np.random.default_rng(20261021)
    pool = rng.permutation(family_pool('test'))
    n,m = cfg['eval']['ordinary_families'],cfg['eval']['long_families']
    if n+m>len(pool):
        raise ValueError('Requested more disjoint test families than the held-out split contains.')
    rows = []
    for cohort,selected,delays,trees in [('ordinary',pool[:n],cfg['eval']['ordinary_delays'],cfg['eval']['tree_ordinary_families']),
                                       ('long',pool[n:n+m],cfg['eval']['long_delays'],cfg['eval']['tree_long_families'])]:
        ranked = sorted(map(int,selected),key=lambda f:sha256_text(f'tree:{f}'))
        tree_set = set(ranked[:trees])
        for i,f in enumerate(selected):
            d = delays[i%len(delays)]
            local = np.random.default_rng(int(f)+991000)
            rows.append(dict(family=int(f),cohort=cohort,delay=int(d),waiting=local.choice([A,B],size=d).tolist(),tree=int(f) in tree_set))
    plan = dict(families=rows,selected_without_model_outputs=True,hash_selection='SHA256(tree:family)',seed=20261021)
    write_json(path,plan)
    return plan


def sha256_text(text):
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


@torch.no_grad()
def cache_correctness(cfg,run,seed,model,device):
    rng = np.random.default_rng(seed+51000)
    logit_errors,prob_errors,self_errors,donor_errors,residual_negative = [],[],[],[],[]
    count = cfg['eval']['cache_checks']
    for offset in range(0,count,32):
        n = min(32,count-offset)
        hs = []
        for _ in range(n):
            fam = int(rng.choice(family_pool('analysis_val')))
            hs.append(make_root(fam,int(rng.integers(2)),rng.choice([A,B],size=int(rng.integers(13))).tolist()))
        state = prefill(model,hs,device)
        query = torch.full((n,),QUERY,device=device)
        cached = model(query[:,None],cache=state.cache)
        ids,mask = pad_histories([h+[QUERY] for h in hs])
        full = model(torch.as_tensor(ids,device=device),torch.as_tensor(mask,device=device))
        positions = torch.tensor([len(h) for h in hs],device=device)
        expected = full.logits[torch.arange(n,device=device),positions]
        logit_errors.extend((cached.logits[:,-1]-expected).abs().amax(-1).cpu().tolist())
        prob_errors.extend(tv(cached.logits[:,-1].softmax(-1),expected.softmax(-1)).cpu().tolist())
        identical = exchange(state.cache,state.cache,True,True)
        replay = model(query[:,None],cache=identical).logits[:,-1].softmax(-1)
        self_errors.extend(tv(replay,cached.logits[:,-1].softmax(-1)).cpu().tolist())
        donor_h = [h[:17]+[2 if h[17]==1 else 1]+h[18:] for h in hs]
        donor = prefill(model,donor_h,device)
        patched = exchange(state.cache,donor.cache,True,True)
        donor_p = model(query[:,None],cache=donor.cache).logits[:,-1].softmax(-1)
        hybrid_p = model(query[:,None],cache=patched).logits[:,-1].softmax(-1)
        donor_errors.extend(tv(donor_p,hybrid_p).cpu().tolist())
        edited=TeacherState(state.cache,donor.probs,donor.features,state.stages,state.positions)
        forced=teacher_step(model,edited,query)
        residual_negative.extend(tv(forced.probs,cached.logits[:,-1].softmax(-1)).cpu().tolist())
    result = dict(logit_max=max(logit_errors),tv_max=max(prob_errors),self_swap_tv_max=max(self_errors),
                  full_donor_tv_max=max(donor_errors),prefixes=count,precision='FP32',
                  residual_only_forced_tv_max=max(residual_negative),
                  passed=max(logit_errors)<=cfg['gates']['cache_logit_error'] and
                  max(prob_errors+self_errors+donor_errors+residual_negative)<=cfg['gates']['cache_tv'])
    write_json(Path(run)/f'seed_{seed}'/'teacher'/'cache_checks.json',result)
    return result


@torch.no_grad()
def teacher_quality(cfg,run,seed,model,device):
    rng = np.random.default_rng(seed+52000)
    waiting,answer,impossible,resets = [],[],[],[]
    n = cfg['train']['validation_episodes']
    for start in range(0,n,64):
        histories = [episode(rng,split='analysis_val') for _ in range(min(64,n-start))]
        ids,mask = pad_histories(histories)
        output = model(torch.as_tensor(ids,device=device),torch.as_tensor(mask,device=device))
        for i,h in enumerate(histories):
            for length in range(18,len(h)):
                state = oracle_state(h[:length])
                if state.stage==Stage.STOP:
                    continue
                p = output.logits[i,length-1].softmax(-1).cpu().numpy()
                exact = np.asarray(list(map(float,oracle_distribution(h[:length]))))
                impossible.append(float(p[exact==0].sum()))
                if state.stage==Stage.WAIT:
                    waiting.append(float(tv(p,exact)))
                elif state.stage==Stage.ANSWER:
                    answer.append(abs(float(p[ONE])-float(exact[ONE])))
        # Exact matched reset pairs at supported, equal-length roots.
        fam = int(rng.choice(family_pool('analysis_val')))
        h0,h1 = [make_root(fam,c,[A,B,A,B],True)+[QUERY] for c in (0,1)]
        p = prefill(model,[h0,h1],device).probs
        resets.append(float(tv(p[0],p[1])))
    result = dict(waiting_tv=summary(waiting),query_mae=summary(answer),impossible_mass=summary(impossible),
                  reset_pair_tv=summary(resets),split='analysis_val',not_final_test=True)
    result['passed'] = (result['waiting_tv']['mean'] is not None and result['query_mae']['mean'] is not None and
                       result['waiting_tv']['mean']<=cfg['gates']['waiting_oracle_tv'] and
                       result['query_mae']['mean']<=cfg['gates']['query_oracle_mae'] and
                       result['query_mae']['p95']<=cfg['gates']['query_oracle_p95'] and
                       result['impossible_mass']['mean']<=cfg['gates']['impossible_mass'] and
                       result['reset_pair_tv']['mean']<=cfg['gates']['reset_tv'])
    write_json(Path(run)/f'seed_{seed}'/'teacher'/'quality.json',result)
    return result


def verify_selection(run,seed):
    root = Path(run)/f'seed_{seed}'
    lock = read_json(root/'selection_lock.json')
    for path,digest in lock['files'].items():
        if sha256(root/path)!=digest:
            raise RuntimeError(f'Selected fitting artifact changed after sealing: {path}')


def initial_z(surrogate,prep,features,source,device):
    return surrogate.encode(torch.as_tensor(prep.transform(features[source].detach().cpu().numpy()),device=device))


@torch.no_grad()
def rollout(model,state,surrogate,z,tokens):
    stages,positions = state.stages,state.positions
    current = state
    for token in tokens:
        supplied = torch.full_like(stages,int(token))
        z = surrogate.update(z,supplied,stages,positions)
        stages,positions = stage_tensor(stages,supplied),positions+1
        current = teacher_step(model,current,supplied)
    return current,z,surrogate.predict(z,stages,positions)


@torch.no_grad()
def surrogate_continuation(surrogate,z,stages,positions,tokens):
    for token in tokens:
        supplied = torch.full_like(stages,int(token))
        z = surrogate.update(z,supplied,stages,positions)
        stages,positions = stage_tensor(stages,supplied),positions+1
    return z,surrogate.predict(z,stages,positions)


@torch.no_grad()
def surrogate_string_probability(surrogate,z,stages,positions,tokens):
    mass=torch.ones(len(z),device=z.device)
    for token in tokens:
        mass=mass*surrogate.predict(z,stages,positions)[:,int(token)]
        supplied=torch.full_like(stages,int(token))
        z=surrogate.update(z,supplied,stages,positions)
        stages,positions=stage_tensor(stages,supplied),positions+1
    return mass


@torch.no_grad()
def evaluate_seed(cfg,run,seed,model,device):
    from .features import load_preprocessor
    from .fit import load_surrogate
    verify_selection(run,seed)
    directory = Path(run)/f'seed_{seed}'/'evaluation'
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'complete.json').exists():
        return
    plan = test_plan(cfg,run)['families']
    predictors = {}
    for source in cfg['analysis']['sources']:
        prep = load_preprocessor(run,seed,source)
        for family in cfg['analysis']['families']:
            predictors[(source,family)] = (prep,load_surrogate(run,seed,source,family,device))
    pairs,ordinary,joints,effects,updates,geometry,strings,timelines = [],[],[],[],[],[],[],[]
    ordinary_arrays=[[],[]]
    effect_arrays=[[],[]]
    joint_arrays=[[],[]]
    trajectories,trajectory_labels = [],[]
    rng = np.random.default_rng(seed+61000)
    tree_batch = 8
    for cohort in ('ordinary','long'):
        for delay in sorted(set(row['delay'] for row in plan if row['cohort']==cohort)):
            group = [r for r in plan if r['cohort']==cohort and r['delay']==delay]
            for start in range(0,len(group),tree_batch):
                records = group[start:start+tree_batch]
                n = len(records)
                for reset in (False,True):
                    histories = [[make_root(r['family'],clue,r['waiting'],reset) for r in records] for clue in (0,1)]
                    states = [prefill(model,h,device) for h in histories]
                    if cohort=='ordinary' and start==0 and not reset and not (directory/'audit_cache_pair.pt').exists():
                        audit=[]
                        for state in states:
                            audit.append(dict(keys=[k[:1].detach().cpu() for k in state.cache.keys],
                                              values=[v[:1].detach().cpu() for v in state.cache.values],
                                              valid=state.cache.valid[:1].cpu(),probs=state.probs[:1].cpu()))
                        save_torch(directory/'audit_cache_pair.pt',dict(seed=seed,histories=[h[0] for h in histories],states=audit))
                    root_match = tv(states[0].probs,states[1].probs).cpu().numpy()
                    tree_ix = [i for i,r in enumerate(records) if r['tree']]
                    tree_targets = {}
                    if tree_ix:
                        ix = torch.tensor(tree_ix,device=device)
                        for clue in (0,1):
                            for horizon in cfg['eval']['tree_horizons']:
                                tree_targets[(clue,horizon)] = teacher_tree(model,states[clue].select(ix),horizon)
                    original_answers = [teacher_step(model,s,torch.full((n,),QUERY,device=device)) for s in states]
                    for i,r in enumerate(records):
                        pairs.append(dict(seed=seed,family=r['family'],cohort=cohort,delay=delay,reset=reset,
                                          current_tv=float(root_match[i]),matched=bool(root_match[i]<=cfg['eval']['matching_tv']),
                                          query_gap=float(abs(original_answers[1].probs[i,ONE]-original_answers[0].probs[i,ONE])),
                                          reset_query_tv=float(tv(original_answers[0].probs[i],original_answers[1].probs[i])) if reset else None,
                                          teacher_query_mae=float(np.mean([abs(float(original_answers[c].probs[i,ONE])-(.5 if reset else .2 if c==0 else .8)) for c in (0,1)]))))
                    encoded = {}
                    for (source,family),(prep,surrogate) in predictors.items():
                        encoded[(source,family)] = [initial_z(surrogate,prep,s.features,source,device) for s in states]
                        density_tree_errors={}
                        for clue,state in enumerate(states):
                            z = encoded[(source,family)][clue]
                            pred = surrogate.predict(z,state.stages,state.positions)
                            pred_np=pred.cpu().numpy()
                            target_np=state.probs.cpu().numpy()
                            root_errors=tv(pred_np,target_np)
                            for i,r in enumerate(records):
                                ordinary.append(dict(seed=seed,source=source,family_model=family,family=r['family'],
                                                     cohort=cohort,delay=delay,reset=reset,clue=clue,matched=bool(root_match[i]<=.01),
                                                     root_tv=float(root_errors[i]),distribution_row=len(ordinary)))
                                ordinary_arrays[0].append(target_np[i]);ordinary_arrays[1].append(pred_np[i])
                            if tree_ix:
                                ix = torch.tensor(tree_ix,device=device)
                                for horizon in cfg['eval']['tree_horizons']:
                                    prediction = surrogate_tree(surrogate,z[ix],state.stages[ix],state.positions[ix],horizon)
                                    target = tree_targets[(clue,horizon)]
                                    err = tv(prediction,target).cpu().numpy()
                                    prediction_np,target_np=prediction.cpu().numpy(),target.cpu().numpy()
                                    if family=='density2' and horizon==3:
                                        density_tree_errors[clue]=err
                                    for j,i in enumerate(tree_ix):
                                        r = records[i]
                                        joints.append(dict(seed=seed,source=source,family_model=family,family=r['family'],cohort=cohort,
                                                           delay=delay,reset=reset,clue=clue,horizon=horizon,matched=bool(root_match[i]<=.01),
                                                           joint_tv=float(err[j]),mass_error=float(abs(prediction_np[j].sum()-1)),distribution_row=len(joints)))
                                        joint_arrays[0].append(target_np[j]);joint_arrays[1].append(prediction_np[j])
                        if family=='density2':
                            dist = surrogate.distance(*encoded[(source,family)]).cpu().numpy()
                            for j,i in enumerate(tree_ix):
                                future_distance=float(tv(tree_targets[(0,3)][j],tree_targets[(1,3)][j]))
                                bound=float(dist[i]+density_tree_errors[0][j]+density_tree_errors[1][j])
                                geometry.append(dict(seed=seed,source=source,family=records[i]['family'],cohort=cohort,delay=delay,
                                                     reset=reset,trace_distance=float(dist[i]),current_pair_tv=float(root_match[i]),
                                                     future3_pair_tv=future_distance,triangle_upper=bound,
                                                     bound_violation=max(0.,future_distance-bound)))
                    # A representative paired trajectory, with matched stage/position labels.
                    if cohort=='ordinary' and start==0:
                        prep,dm=predictors[('cache_summary','density2')]
                        for clue,initial in enumerate(states):
                            reference=initial
                            zz=encoded[('cache_summary','density2')][clue].clone()
                            ss,pp=initial.stages,initial.positions
                            for offset,token in enumerate([None]+[A,B,A,B,A,B,QUERY]):
                                if token is not None:
                                    xx=torch.full_like(ss,token)
                                    zz=dm.update(zz,xx,ss,pp)
                                    ss,pp=stage_tensor(ss,xx),pp+1
                                    reference=teacher_step(model,reference,xx)
                                predicted=dm.predict(zz,ss,pp)
                                timelines.append(dict(seed=seed,family=records[0]['family'],delay=delay,reset=reset,clue=clue,
                                                      offset=offset,token='root' if token is None else TOKENS[token],stage=int(ss[0]),
                                                      actual_p1=float(reference.probs[0,ONE]),predicted_p1=float(predicted[0,ONE]),
                                                      full_distribution_tv=float(tv(predicted[0],reference.probs[0]))))
                    # Supported random eight-token observations: one initial encoding, no refresh.
                    for clue,state in enumerate(states):
                        continuations = [sample_from_prefix(rng,h,8) for h in histories[clue]]
                        current = state
                        zstates = {key:value[clue].clone() for key,value in encoded.items()}
                        stages,positions = state.stages,state.positions
                        for k in range(1,9):
                            tokens = torch.tensor([c[k-1] if k<=len(c) else END for c in continuations],device=device)
                            for key,(prep,surrogate) in predictors.items():
                                zstates[key] = surrogate.update(zstates[key],tokens,stages,positions)
                            stages,positions = stage_tensor(stages,tokens),positions+1
                            current = teacher_step(model,current,tokens)
                            for (source,family),(prep,surrogate) in predictors.items():
                                z = zstates[(source,family)]
                                p = surrogate.predict(z,stages,positions)
                                fresh = initial_z(surrogate,prep,current.features,source,device)
                                # Fresh encodings are diagnostic targets, never assigned to z.
                                dd = surrogate.distance(z,fresh).cpu().numpy()
                                output_errors=tv(p,current.probs).cpu().numpy()
                                for i,r in enumerate(records):
                                    if int(stages[i])==Stage.STOP:
                                        continue
                                    if k in cfg['eval']['rollout_steps']:
                                        updates.append(dict(seed=seed,source=source,family_model=family,family=r['family'],cohort=cohort,
                                                            delay=delay,reset=reset,clue=clue,steps=k,output_tv=float(output_errors[i]),
                                                            state_distance=float(dd[i]),matched=bool(root_match[i]<=.01),
                                                            supplied_tokens=' '.join(TOKENS[x] for x in continuations[i][:k]),
                                                            stage=int(stages[i]),position=int(positions[i])))
                                    if source=='cache_summary' and family=='density2' and len(trajectories)<1500:
                                        trajectories.append(z[i].cpu().numpy())
                                        trajectory_labels.append([r['family'],clue,int(reset),k,int(stages[i]),int(positions[i])])
                    # Main causal endpoints, both donor directions, four prespecified arms.
                    for recipient_clue in (0,1):
                        recipient,donor = states[recipient_clue],states[1-recipient_clue]
                        for cont_text in cfg['eval']['continuations']:
                            continuation_tokens = [TOKENS.index(x) for x in cont_text]
                            baseline = recipient
                            baseline_mass=torch.ones(n,device=device)
                            for x in continuation_tokens:
                                baseline_mass=baseline_mass*baseline.probs[:,x]
                                baseline = teacher_step(model,baseline,torch.full((n,),x,device=device))
                            base_pred = {}
                            for key,(prep,surrogate) in predictors.items():
                                _,p = surrogate_continuation(surrogate,encoded[key][recipient_clue].clone(),
                                                           recipient.stages,recipient.positions,continuation_tokens)
                                base_pred[key] = p
                                mass=surrogate_string_probability(surrogate,encoded[key][recipient_clue].clone(),
                                                                  recipient.stages,recipient.positions,continuation_tokens)
                                for i,r in enumerate(records):
                                    strings.append(dict(seed=seed,source=key[0],family_model=key[1],family=r['family'],cohort=cohort,
                                                        delay=delay,reset=reset,clue=recipient_clue,continuation=cont_text,
                                                        teacher_probability=float(baseline_mass[i]),surrogate_probability=float(mass[i]),
                                                        absolute_error=float(abs(mass[i]-baseline_mass[i]))))
                            for arm,early,current_flag in [('00',False,False),('10',True,False),('01',False,True),('11',True,True)]:
                                patched_cache = exchange(recipient.cache,donor.cache,early,current_flag)
                                patched_features = dict(recipient.features)
                                patched_features['cache_summary'] = cache_features(patched_cache)
                                patched = TeacherState(patched_cache,recipient.probs,patched_features,recipient.stages,recipient.positions)
                                # Compute all frozen surrogate predictions before replaying
                                # the edited teacher or calculating any prediction errors.
                                predicted_distributions={}
                                for (source,family),(prep,surrogate) in predictors.items():
                                    z=initial_z(surrogate,prep,patched_features,source,device)
                                    _,prediction=surrogate_continuation(surrogate,z,patched.stages,patched.positions,continuation_tokens)
                                    predicted_distributions[(source,family)]=prediction
                                reference = patched
                                for x in continuation_tokens:
                                    reference = teacher_step(model,reference,torch.full((n,),x,device=device))
                                actual = reference.probs[:,ONE]-baseline.probs[:,ONE]
                                reference_np=reference.probs.cpu().numpy()
                                baseline_np=baseline.probs.cpu().numpy()
                                actual_np=actual.cpu().numpy()
                                impossible_np=impossible_mass(reference.probs,reference.stages).cpu().numpy()
                                actual_tvs=tv(reference_np,baseline_np)
                                log_ratios=np.log(np.maximum(reference_np[:,ONE],1e-12)/np.maximum(reference_np[:,1],1e-12))-np.log(np.maximum(baseline_np[:,ONE],1e-12)/np.maximum(baseline_np[:,1],1e-12))
                                for (source,family),(prep,surrogate) in predictors.items():
                                    prediction=predicted_distributions[(source,family)]
                                    predicted = prediction[:,ONE]-base_pred[(source,family)][:,ONE]
                                    prediction_np=prediction.cpu().numpy()
                                    base_np=base_pred[(source,family)].cpu().numpy()
                                    predicted_log=np.log(np.maximum(prediction_np[:,ONE],1e-12)/np.maximum(prediction_np[:,1],1e-12))-np.log(np.maximum(base_np[:,ONE],1e-12)/np.maximum(base_np[:,1],1e-12))
                                    predicted_np=predicted.cpu().numpy()
                                    future_errors=tv(prediction_np,reference_np)
                                    for i,r in enumerate(records):
                                        a,pd = float(actual_np[i]),float(predicted_np[i])
                                        effects.append(dict(seed=seed,source=source,family_model=family,family=r['family'],cohort=cohort,
                                                            delay=delay,reset=reset,recipient_clue=recipient_clue,arm=arm,continuation=cont_text,
                                                            matched=bool(root_match[i]<=.01),actual_effect=a,predicted_effect=pd,
                                                            effect_error=abs(a-pd),donor_directed_effect=a*(1 if recipient_clue==0 else -1),
                                                            sign_correct=bool(np.sign(a)==np.sign(pd)),
                                                            actual_p1=float(reference_np[i,ONE]),predicted_p1=float(prediction_np[i,ONE]),
                                                            baseline_p1=float(baseline_np[i,ONE]),future_tv=float(future_errors[i]),
                                                            actual_distribution_tv=float(actual_tvs[i]),log_ratio_change=float(log_ratios[i]),
                                                            predicted_log_ratio_change=float(predicted_log[i]),
                                                            predicted_distribution_tv=float(tv(prediction_np[i],base_np[i])),
                                                            impossible_mass=float(impossible_np[i]),distribution_row=len(effects),
                                                            state_access='edited_cache' if source=='cache_summary' else 'unchanged_recipient_residual'))
                                        effect_arrays[0].append(reference_np[i]);effect_arrays[1].append(prediction_np[i])
                progress(f'seed={seed} evaluated {cohort} delay={delay}: {min(start+tree_batch,len(group))}/{len(group)} families')
    for name,rows in [('pairs',pairs),('ordinary',ordinary),('joints',joints),('effects',effects),('updates',updates),('geometry',geometry),('strings',strings),('timeline',timelines)]:
        write_csv(directory/f'{name}.csv',rows)
    for name,arrays in [('ordinary',ordinary_arrays),('effects',effect_arrays)]:
        np.savez_compressed(directory/f'{name}_distributions.npz',teacher=np.asarray(arrays[0]),surrogate=np.asarray(arrays[1]),vocabulary=np.asarray(TOKENS))
    offsets=np.cumsum([0]+[len(x) for x in joint_arrays[0]])
    np.savez_compressed(directory/'joint_distributions.npz',teacher=np.concatenate(joint_arrays[0]),
                        surrogate=np.concatenate(joint_arrays[1]),offsets=offsets,vocabulary=np.asarray(TOKENS))
    np.savez_compressed(directory/'density_trajectories.npz',coordinates=np.asarray(trajectories),labels=np.asarray(trajectory_labels),
                        label_columns=np.asarray(['family','clue','reset','offset','stage','position']))
    evaluate_natural(cfg,run,seed,model,predictors,device)
    benchmark_predictors(cfg,run,seed,predictors,device)
    write_interactions_and_calibration(directory,effects,geometry)
    write_json(directory/'complete.json',dict(profile=cfg['profile'],seed=seed,records={
               'pairs':len(pairs),'ordinary':len(ordinary),'joints':len(joints),'effects':len(effects),'updates':len(updates)},
               all_eight_tokens_scored=True,teacher_refreshed_surrogate=False,interventions_used_for_fitting=False))


def write_interactions_and_calibration(directory,effects,geometry):
    import pandas as pd
    df=pd.DataFrame(effects)
    keys=['source','family_model','family','cohort','delay','reset','recipient_clue','continuation']
    wide=df.pivot(index=keys,columns='arm',values='actual_p1')
    wide['interaction']=wide['11']-wide['10']-wide['01']+wide['00']
    wide.reset_index().to_csv(directory/'interactions.csv',index=False)
    rows=[]
    for (source,family),group in df.groupby(['source','family_model']):
        for bin_index in range(10):
            sel=group[(group.predicted_p1>=bin_index/10)&(group.predicted_p1<(bin_index+1)/10+1e-12)]
            if len(sel):
                rows.append(dict(source=source,family_model=family,bin=bin_index,count=len(sel),
                                 mean_actual=sel.actual_p1.mean(),mean_predicted=sel.predicted_p1.mean(),
                                 mean_absolute_error=np.abs(sel.actual_p1-sel.predicted_p1).mean()))
    write_csv(directory/'calibration.csv',rows)
    g=pd.DataFrame(geometry)
    stats=[]
    for (source,cohort,delay,reset),group in g.groupby(['source','cohort','delay','reset']):
        correlation=group.trace_distance.corr(group.future3_pair_tv) if len(group)>1 else None
        stats.append(dict(source=source,cohort=cohort,delay=int(delay),reset=bool(reset),count=len(group),
                          pearson=correlation,max_triangle_bound_violation=group.bound_violation.max()))
    write_json(directory/'geometry_diagnostics.json',stats)


@torch.no_grad()
def evaluate_natural(cfg,run,seed,model,predictors,device):
    """Separate generator-weighted held-out observations from controlled long-delay stress tests."""
    import joblib
    rng = np.random.default_rng(seed+62000)
    rows,diagnostic_rows = [],[]
    count = cfg['eval']['natural_episodes']
    histories=[episode(rng,split='test') for _ in range(count)]
    # Every observed nonterminal post-clue position from naturally drawn episodes.
    # This is token-frequency weighting, not uniform-over-controlled-delays weighting.
    observations=[(i,h[:length]) for i,h in enumerate(histories) for length in range(18,len(h))]
    for start in range(0,len(observations),64):
        chunk=observations[start:start+64]
        roots=[h for _,h in chunk]
        state = prefill(model,roots,device)
        for (source,family),(prep,surrogate) in predictors.items():
            z = initial_z(surrogate,prep,state.features,source,device)
            p = surrogate.predict(z,state.stages,state.positions)
            p_np=p.cpu().numpy();teacher_np=state.probs.cpu().numpy()
            errors=tv(p_np,teacher_np)
            for i,h in enumerate(roots):
                oracle = np.array(list(map(float,oracle_distribution(h))))
                rows.append(dict(seed=seed,source=source,family_model=family,episode=chunk[i][0],
                                 target_tv=float(errors[i]),
                                 teacher_oracle_tv=float(tv(teacher_np[i],oracle)),
                                 stage=int(state.stages[i]),position=len(h)-1))
        for source in cfg['analysis']['sources']:
            prep = next(p for (s,f),(p,m) in predictors.items() if s==source)
            raw = state.features[source].cpu().numpy()
            target = np.array([float(oracle_state(h).belief) for h in roots])
            for mode in ('native','processed'):
                reg = joblib.load(Path(run)/f'seed_{seed}'/'diagnostics'/f'ridge_{source}_{mode}.pkl')
                x = prep.transform(raw) if mode=='processed' else (raw-prep.mean)/prep.scale
                for i,pred in enumerate(reg.predict(x)):
                    diagnostic_rows.append(dict(seed=seed,source=source,mode=mode,episode=chunk[i][0],stage=int(state.stages[i]),
                                                belief=float(target[i]),prediction=float(pred),absolute_error=float(abs(pred-target[i]))))
    write_csv(Path(run)/f'seed_{seed}'/'evaluation'/'natural.csv',rows)
    write_csv(Path(run)/f'seed_{seed}'/'diagnostics'/'held_out_posterior.csv',diagnostic_rows)


@torch.no_grad()
def benchmark_predictors(cfg,run,seed,predictors,device):
    rows = []
    for (source,family),(prep,model) in predictors.items():
        for size in cfg['eval']['benchmark_batches']:
            x=torch.zeros(size,128,device=device)
            z = model.encode(x)
            stages = torch.zeros(size,dtype=torch.long,device=device)
            positions = torch.full_like(stages,22)
            tokens=torch.full_like(stages,QUERY)
            operations={'encode':lambda:model.encode(x),'predict':lambda:model.predict(z,stages,positions),
                        'update':lambda:model.update(z,tokens,stages,positions)}
            for operation,call in operations.items():
                for _ in range(3):
                    call()
                synchronize(device)
                base=torch.cuda.memory_allocated(device) if device.type=='cuda' else None
                if device.type=='cuda':
                    torch.cuda.reset_peak_memory_stats(device)
                tick = time.perf_counter()
                for _ in range(cfg['eval']['benchmark_repeats']):
                    call()
                synchronize(device)
                seconds = (time.perf_counter()-tick)/cfg['eval']['benchmark_repeats']
                peak=torch.cuda.max_memory_allocated(device)-base if device.type=='cuda' else None
                rows.append(dict(source=source,family_model=family,batch_size=size,operation=operation,seconds_per_call=seconds,
                             numeric_state_bytes=(1 if family=='classical2' else 3)*4,
                             materialized_density_bytes=64 if family=='density2' else None,
                             actual_working_peak_extra_vram_bytes=peak,stage_position_bytes=16,counts=model.counts()))
            raw=np.zeros((size,len(prep.mean)),np.float32)
            tick=time.perf_counter()
            for _ in range(cfg['eval']['benchmark_repeats']):
                prep.transform(raw)
            rows.append(dict(source=source,family_model=family,batch_size=size,operation='preprocess_cpu',
                             seconds_per_call=(time.perf_counter()-tick)/cfg['eval']['benchmark_repeats']))
    write_csv(Path(run)/f'seed_{seed}'/'evaluation'/'benchmark.csv',rows)
