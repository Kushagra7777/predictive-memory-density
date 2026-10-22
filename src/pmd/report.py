"""Predeclared PDF gates. Missing data and integration runs never receive a green signal."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .io import read_json,write_csv,write_json
from .metrics import family_bootstrap


def mean(frame,column):
    return None if len(frame)==0 else float(frame[column].mean())


def upper(frame,column):
    return None if len(frame)==0 else float(frame[column].quantile(.95))


def gate(name,conditions,values,criteria):
    conditions = list(conditions)
    passed = None if any(x is None for x in conditions) else bool(all(conditions))
    return dict(name=name,passed=passed,status='INSUFFICIENT_DATA' if passed is None else 'PASS' if passed else 'FAIL',
                values=values,criteria=criteria)


def le(value,threshold):
    return None if value is None or not np.isfinite(value) else value<=threshold


def ge(value,threshold):
    return None if value is None or not np.isfinite(value) else value>=threshold


def frame(path):
    try:
        return pd.read_csv(path)
    except (FileNotFoundError,pd.errors.EmptyDataError):
        return pd.DataFrame()


def report_seed(cfg,run,seed):
    root = Path(run)/f'seed_{seed}'
    evaldir = root/'evaluation'
    g = cfg['gates']
    pairs,ordinary,joints,effects,updates = [frame(evaldir/f'{name}.csv') for name in ['pairs','ordinary','joints','effects','updates']]
    quality = read_json(root/'teacher'/'quality.json')
    cache = read_json(root/'teacher'/'cache_checks.json')
    clean = pairs[(pairs.cohort=='ordinary')&~pairs.reset]
    reset_pairs = pairs[(pairs.cohort=='ordinary')&pairs.reset]
    long_pairs = pairs[(pairs.cohort=='long')&~pairs.reset]
    def primary(df):
        return df[(df.source=='cache_summary')&(df.family_model=='density2')]
    obs = primary(ordinary)
    obs = obs[(obs.cohort=='ordinary')&~obs.reset]
    joint = primary(joints)
    joint = joint[(joint.cohort=='ordinary')&~joint.reset&(joint.horizon==3)]
    upd = primary(updates) if len(updates) else updates
    upd = upd[(upd.cohort=='ordinary')&~upd.reset] if len(upd) else upd
    # CSV numeric conversion turns '01' into 1; normalize once for all reports.
    effects['arm'] = effects.arm.astype(str).str.zfill(2)
    write_endpoint_summaries(cfg,root,ordinary,joints,effects,updates)
    fx = primary(effects)
    fx = fx[(fx.cohort=='ordinary')&fx.matched&fx.arm.isin(['10','01'])]
    mainfx,resetfx = fx[~fx.reset],fx[fx.reset]
    big = mainfx[mainfx.actual_effect.abs()>=g['effect_sign_minimum']]
    gates = []
    gates.append(gate('generator',[read_json(Path(run)/'generator'/'oracle_checks.json')['passed']],{},'Exact identities and recursive reset argument; oracle error <1e-12'))
    gates.append(gate('cache',[cache['passed']],cache,'FP32 logit error <=1e-5; cached/self/full-donor TV <=1e-6'))
    gates.append(gate('transformer_learning',[quality['passed']],quality,'Waiting TV <=.01; query MAE <=.03/p95<=.08; impossible mass<=.005; reset TV<=.03'))
    matched = mean(clean,'matched')
    gap = mean(clean,'query_gap')
    gates.append(gate('pair_validity',[ge(matched,g['matched_fraction']),ge(gap,g['query_gap'])],
                      dict(matched_fraction=matched,query_gap=gap),'At least 90% full-distribution matches; mean query gap >=.50; report all pairs'))
    reset_tv = mean(reset_pairs,'reset_query_tv')
    gates.append(gate('reset_learning',[le(reset_tv,g['reset_tv'])],dict(reset_query_tv=reset_tv),'Mean reset-pair query TV <=.03'))
    root_error,joint_error = mean(obs,'root_tv'),mean(joint,'joint_tv')
    gates.append(gate('ordinary_state',[le(root_error,g['root_tv']),le(joint_error,g['joint3_tv'])],
                      dict(root_tv=root_error,joint3_tv=joint_error),'Root TV <=.02; complete joint-three-token TV <=.05'))
    u1 = upd[upd.steps==1] if len(upd) else upd
    u4 = upd[upd.steps==4] if len(upd) else upd
    u8 = upd[upd.steps==8] if len(upd) else upd
    vals = dict(distance1=mean(u1,'state_distance'),distance8=mean(u8,'state_distance'),
                output_tv4=mean(u4,'output_tv'),output_tv8=mean(u8,'output_tv'))
    gates.append(gate('density_updates',[le(vals['distance1'],g['update_distance1']),le(vals['distance8'],g['update_distance8']),
                                        le(vals['output_tv4'],g['rollout_tv4']),le(vals['output_tv8'],g['rollout_tv8'])],
                      vals,'Distance one/eight <=.05/.10; output TV four/eight <=.03/.05; no refreshes'))
    lo = primary(ordinary)
    lo = lo[(lo.cohort=='long')&~lo.reset]
    lj = primary(joints)
    lj = lj[(lj.cohort=='long')&~lj.reset&(lj.horizon==3)]
    longfx=primary(effects)
    longfx=longfx[(longfx.cohort=='long')&~longfx.reset&(longfx.arm=='00')]
    vals = dict(teacher_query_mae=mean(long_pairs,'teacher_query_mae'),root_tv=mean(lo,'root_tv'),
                conditional_future_tv=mean(longfx,'future_tv'),joint3_tv=mean(lj,'joint_tv'))
    gates.append(gate('longer_delays',[le(vals['teacher_query_mae'],g['long_teacher_query_mae']),le(vals['root_tv'],g['long_output_tv']),
                                     le(vals['conditional_future_tv'],g['long_output_tv']),
                                     le(vals['joint3_tv'],g['long_joint3_tv'])],vals,'Delays 24/32: teacher MAE <=.05; surrogate root/conditional TV<=.05; joint3 TV<=.07'))
    arm_stats = {}
    arm_pass = []
    for arm in ('10','01'):
        arm_rows = mainfx[mainfx.arm==arm].to_dict('records')
        interval = family_bootstrap(arm_rows,'donor_directed_effect',samples=cfg['eval']['bootstrap_samples'],statistic='median')
        arm_stats[arm] = interval
        arm_pass.append(None if interval['lower'] is None else interval['estimate']>=g['causal_effect_median'] and interval['lower']>g['causal_effect_lower'])
    reset_effect = None if not len(resetfx) else float(resetfx.actual_effect.abs().mean())
    disruption = {f'{arm}_reset_{int(reset)}':mean(fx[(fx.arm==arm)&(fx.reset==reset)],'impossible_mass')
                  for arm in ('10','01') for reset in (False,True)}
    any_arm = None if all(x is None for x in arm_pass) else any(x is True for x in arm_pass)
    gates.append(gate('causal_memory',[any_arm,le(reset_effect,g['reset_effect_mae']),
                                      *[le(v,g['patched_impossible_mass']) for v in disruption.values()]],
                      dict(arms=arm_stats,reset_absolute_effect=reset_effect,impossible_mass=disruption),
                      'At least one prespecified partial arm: median >=.20 and family CI lower>.10; reset <=.03; disruption <=.01'))
    vals = dict(effect_mae=mean(mainfx,'effect_error'),sign_accuracy=mean(big,'sign_correct'),future_tv=mean(mainfx,'future_tv'),
                sign_scored_cases=len(big))
    gates.append(gate('intervention_prediction',[le(vals['effect_mae'],g['effect_mae']),ge(vals['sign_accuracy'],g['effect_sign_accuracy']),
                                               le(vals['future_tv'],g['patched_future_tv'])],vals,
                      'Signed-effect MAE<=.05; sign accuracy>=90% when actual |effect|>=.10; future TV<=.05'))
    comparisons = []
    paired_conditions = []
    for baseline in ('classical4','vector3'):
        for metric,data,keys in [('joint_tv',joints,['family','cohort','delay','reset','clue','horizon']),
                                 ('effect_error',effects,['family','cohort','delay','reset','recipient_clue','arm','continuation'])]:
            sub = data[(data.source=='cache_summary')&(data.cohort=='ordinary')&~data.reset&data.matched]
            if metric=='joint_tv':
                sub = sub[sub.horizon==3]
            else:
                sub = sub[sub.arm.isin(['10','01'])]
            left = sub[sub.family_model=='density2'][keys+[metric]].rename(columns={metric:'density_error'})
            right = sub[sub.family_model==baseline][keys+[metric]].rename(columns={metric:'baseline_error'})
            joined = left.merge(right,on=keys,validate='one_to_one')
            joined['improvement'] = joined.baseline_error-joined.density_error
            interval = family_bootstrap(joined.to_dict('records'),'improvement',samples=cfg['eval']['bootstrap_samples'])
            abs_gain = mean(joined,'improvement')
            baseline_error = mean(joined,'baseline_error')
            rel_gain = None if baseline_error is None else abs_gain/max(baseline_error,1e-12)
            comparisons.append(dict(baseline=baseline,metric=metric,absolute_gain=abs_gain,relative_gain=rel_gain,paired_interval=interval))
            paired_conditions.extend([ge(abs_gain,g['density_absolute_gain']),ge(rel_gain,g['density_relative_gain']),
                                      None if interval['lower'] is None else interval['lower']>0])
    gates.append(gate('density_specific_value',paired_conditions,dict(comparisons=comparisons),
                      'Beat BOTH matched classical opponents on joint3 TV AND effect error by >=20% and >=.01; paired CI excludes zero; time-matched fitting'))
    independent = {x['name']:x['passed'] for x in gates}
    # The task/implementation must work before any scientific interpretation is possible.
    prerequisite = all(independent[k] is True for k in ['generator','cache','transformer_learning','pair_validity','reset_learning'])
    memory_signal = prerequisite and independent['causal_memory'] is True
    density_signal = prerequisite and all(x['passed'] is True for x in gates)
    result = dict(seed=seed,profile=cfg['profile'],gates=gates,memory_pilot_pass=memory_signal,
                  density_specific_pass=density_signal,formal=cfg['profile']=='formal')
    write_json(root/'reports'/'gates.json',result)
    return result


def write_endpoint_summaries(cfg,root,ordinary,joints,effects,updates):
    """All-pair and retained-pair estimates, with family intervals and quantiles."""
    from .metrics import summary
    endpoints=[('root_tv',ordinary,'root_tv'),
               ('joint3_tv',joints[joints.horizon==3],'joint_tv'),
               ('partial_effect_error',effects[effects.arm.isin(['10','01'])],'effect_error'),
               ('partial_conditional_tv',effects[effects.arm.isin(['10','01'])],'future_tv')]
    rows=[]
    for name,data,value in endpoints:
        # Native residual inputs are still fully available in raw per-case CSVs.
        data=data[data.source=='cache_summary']
        for (family,cohort,reset),group in data.groupby(['family_model','cohort','reset']):
            for subset in ('all','matched'):
                selected=group if subset=='all' else group[group.matched]
                stats=summary(selected[value].values)
                interval=family_bootstrap(selected.to_dict('records'),value,samples=cfg['eval']['bootstrap_samples'])
                rows.append(dict(endpoint=name,family_model=family,cohort=cohort,reset=bool(reset),subset=subset,
                                 **stats,ci_lower=interval['lower'],ci_upper=interval['upper'],nuisance_families=interval['families']))
    write_csv(root/'reports'/'endpoint_summary.csv',rows)


def build_report(cfg,run):
    directory = Path(run)/'reports'
    directory.mkdir(parents=True,exist_ok=True)
    results = []
    for seed in cfg['seeds']:
        if (Path(run)/f'seed_{seed}'/'evaluation'/'complete.json').exists():
            results.append(report_seed(cfg,run,seed))
    formal = cfg['profile']=='formal' and len(results)==3 and cfg['seeds']==[17,29,43]
    manifest = read_json(Path(run)/'manifest.json')
    formal &= not bool(manifest.get('diagnostic_continue',False))
    memory = formal and all(x['memory_pilot_pass'] for x in results)
    density = formal and all(x['density_specific_pass'] for x in results)
    status = 'DENSITY_PILOT_PASS' if density else 'MEMORY_PILOT_PASS' if memory else 'IMPLEMENTATION_ONLY' if cfg['profile']!='formal' else 'NO_GREEN_SIGNAL'
    training_device=manifest.get('stage_status',{}).get('train',{}).get('device',manifest.get('device',''))
    overview = dict(status=status,profile=cfg['profile'],formal=formal,evaluated_seeds=[x['seed'] for x in results],
                    memory_pilot_pass=memory,density_specific_pass=density,
                    novelty_verified=False,gpu_experiment_performed=training_device.startswith('cuda'),seeds=results)
    write_json(directory/'decision.json',overview)
    rows = [dict(seed=r['seed'],gate=g['name'],status=g['status'],criteria=g['criteria']) for r in results for g in r['gates']]
    write_csv(directory/'gates.csv',rows)
    lines = ['# Same prediction now, different memory later', '',f'Run status: **{status}**',
             f"Profile: `{cfg['profile']}`. Evaluated teacher seeds: {overview['evaluated_seeds']}.", '',
             'This is an empirical pilot decision, not verification of novelty, physical quantum computation, or publication readiness.', '',
             '| Teacher seed | Gate | Result |','|---|---|---|']
    lines += [f"| {r['seed']} | {r['gate']} | {r['status']} |" for r in rows]
    lines += ['', '## Results and plots', '',
              'Per-case CSVs are under `seed_*/evaluation/`. Fitting logs and model selections are separate from held-out evaluations.',
              'Plots are under `plots/`; detailed criteria, measured values, and family-bootstrap intervals are in `seed_*/reports/gates.json`.', '',
              '## Interpretation', '',
              'The primary fair intervention comparison is within the cache-summary input condition. Residual-only tools cannot observe a cache-only edit.',
              'Partial swaps can expose redundant or interacting memory. A donor-directed change is not a percentage of information stored in that block.',
              'Integration-profile results cannot establish the proposed scientific signal. Missing endpoints count as insufficient data, never as passes.', '',
              '## Protocol', '', 'See `docs/PROTOCOL_COVERAGE.md`, `docs/IMPLEMENTATION_DECISIONS.md`, and the source PDF bundled in the repository.']
    (directory/'summary.md').write_text('\n'.join(lines)+'\n')
    return overview
