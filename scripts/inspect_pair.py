#!/usr/bin/env python3
"""Print matrices and predictions for one sealed test family, without fitting anything."""
import argparse
from pathlib import Path
import json
import numpy as np
import torch

from pmd.config import load_config
from pmd.evaluate import prefill,initial_z,teacher_tree,surrogate_tree,teacher_step,test_plan,verify_selection
from pmd.features import load_preprocessor
from pmd.fit import load_surrogate
from pmd.generator import QUERY,TOKENS,make_root,oracle_distribution
from pmd.io import write_json,set_seed,jsonable
from pmd.metrics import tv
from pmd.surrogates import rho_from_bloch
from pmd.train import load_teacher


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--family',type=int,default=None,help='Omit to inspect the first ordinary final-test family.')
    parser.add_argument('--source',choices=['last_residual','all_layers','cache_summary'],default='cache_summary')
    parser.add_argument('--reset',action='store_true')
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    torch.set_num_threads(4);set_seed(0)
    run=Path(args.run_dir).resolve();device=torch.device(args.device)
    cfg=load_config(run/'resolved_config.yaml')
    verify_selection(run,args.seed)
    plan=test_plan(cfg,run)['families']
    record=next((r for r in plan if r['family']==args.family),None) if args.family is not None else next(r for r in plan if r['cohort']=='ordinary')
    if record is None:
        parser.error('Family must belong to this run\'s sealed final-test plan.')
    histories=[make_root(record['family'],c,record['waiting'],args.reset) for c in (0,1)]
    teacher=load_teacher(run,args.seed,device)
    states=[prefill(teacher,[h],device) for h in histories]
    prep=load_preprocessor(run,args.seed,args.source)
    model=load_surrogate(run,args.seed,args.source,'density2',device)
    z=[initial_z(model,prep,s.features,args.source,device) for s in states]
    rho=[rho_from_bloch(v).detach().cpu().numpy()[0] for v in z]
    result=dict(seed=args.seed,family=record['family'],delay=record['delay'],reset=args.reset,source=args.source,
                vocabulary=TOKENS,histories=[' '.join(TOKENS[x] for x in h) for h in histories],
                matrices=[dict(real=v.real,imaginary=v.imag) for v in rho],bloch_coordinates=z,
                matrix_trace_distance=float(model.distance(*z)),
                teacher_current=[s.probs for s in states],current_teacher_tv=float(tv(states[0].probs,states[1].probs)),
                oracle_current=[list(map(float,oracle_distribution(h))) for h in histories],
                same_supplied_query_teacher=[teacher_step(teacher,s,torch.tensor([QUERY],device=device)).probs for s in states],
                note='Matrices are fitted predictive summaries; their entries are not independently identified causal variables.')
    result['joint_futures']=[]
    for horizon in (1,2,3):
        target=[teacher_tree(teacher,s,horizon) for s in states]
        pred=[surrogate_tree(model,v,s.stages,s.positions,horizon) for v,s in zip(z,states)]
        result['joint_futures'].append(dict(horizon=horizon,teacher_pair_tv=tv(*target),surrogate_pair_tv=tv(*pred),
                                          surrogate_error_each=[tv(a,b) for a,b in zip(target,pred)]))
    out=run/'inspections'/f'seed_{args.seed}'/f"family_{record['family']}__reset_{int(args.reset)}__{args.source}.json"
    write_json(out,result)
    print(json.dumps(jsonable(result),indent=2))
    print(f'Saved: {out}')


if __name__=='__main__':
    main()
