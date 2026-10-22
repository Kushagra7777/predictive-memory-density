from __future__ import annotations

import argparse
from datetime import datetime,timezone
from pathlib import Path
import json
import shutil
import sys
import time

import numpy as np
import torch
import yaml

from .config import fingerprint,load_config
from .generator import split_manifest,verify_generator
from .io import append_jsonl,environment,progress,read_json,write_json,set_seed


def initialize(cfg,run,device):
    run=Path(run)
    manifest=run/'manifest.json'
    if manifest.exists():
        if read_json(manifest)['config_sha256']!=fingerprint(cfg):
            raise ValueError('This run has a different resolved configuration. Create a new run directory.')
        return
    baseline=Path(__file__).resolve().parents[2]/'docs'/'protocol_fingerprint.json'
    if cfg['profile']=='formal':
        if not baseline.exists():
            raise RuntimeError('Missing docs/protocol_fingerprint.json: run from the complete editable repository.')
        if read_json(baseline)['resolved_config_sha256']!=fingerprint(cfg):
            raise ValueError('Formal profile differs from the frozen PDF configuration. Declare profile: exploratory for revised experiments.')
    run.mkdir(parents=True,exist_ok=True)
    (run/'resolved_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    write_json(manifest,dict(created_utc=datetime.now(timezone.utc).isoformat(),
                            profile=cfg['profile'],config_sha256=fingerprint(cfg),device=str(device),
                            stage_status={},diagnostic_continue=False))
    write_json(run/'environment'/'environment.json',environment(device))
    write_json(run/'generator'/'split_manifest.json',split_manifest())
    write_json(run/'generator'/'oracle_checks.json',verify_generator())


def mark(run,stage,status,**extra):
    path=Path(run)/'manifest.json'
    obj=read_json(path)
    record=obj['stage_status'].get(stage,{})
    record.update(status=status,utc=datetime.now(timezone.utc).isoformat(),**extra)
    if status!='failed':
        record.pop('error',None)
    obj['stage_status'][stage]=record
    write_json(path,obj)


def doctor(cfg,device):
    from .surrogates import make_surrogate
    from .transformer import Transformer
    report=environment(device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available in this Python environment.')
    x=torch.randn(8,8,device=device)
    y=x@x.T
    report['tensor_test']=bool(torch.isfinite(y).all())
    model=Transformer(cfg['model']).to(device)
    model(torch.zeros(1,18,dtype=torch.long,device=device))
    report['transformer_parameters']=sum(p.numel() for p in model.parameters())
    report['surrogate_counts']={family:make_surrogate(family).counts() for family in cfg['analysis']['families']}
    d=make_surrogate('density2').to(device)
    z=d.encode(torch.randn(4,128,device=device))
    s=torch.zeros(4,dtype=torch.long,device=device)
    p=d.predict(z,s,s+21)
    p.log().sum().backward()
    report['complex_autograd_test']=all(v.grad is None or bool(torch.isfinite(v.grad).all()) for v in d.parameters())
    print(json.dumps(report,indent=2,default=str))
    return report


def execute_stage(stage,cfg,run,seeds,device,args):
    from .train import train_transformer,load_teacher
    from .evaluate import cache_correctness,teacher_quality,evaluate_seed
    from .banks import collect_bank,load_bank
    from .features import fit_preprocessors
    from .fit import fit_all
    from .plots import make_plots
    from .report import build_report
    mark(run,stage,'running',device=str(device))
    write_json(Path(run)/'environment'/f'{stage}.json',environment(device))
    try:
        if stage=='verify':
            write_json(Path(run)/'generator'/'oracle_checks.json',verify_generator())
        elif stage=='smoke':
            seed=cfg['seeds'][0]
            train_transformer(cfg,run,seed,device,smoke=True,resume=args.resume)
            model=load_teacher(run,seed,device,smoke=True)
            # Smoke uses the same parity checks but fewer prefixes, explicitly not scientific results.
            import copy
            small=copy.deepcopy(cfg)
            small['eval']['cache_checks']=min(32,cfg['eval']['cache_checks'])
            checks=cache_correctness(small,run,seed,model,device)
            shutil.copy2(Path(run)/f'seed_{seed}'/'teacher'/'cache_checks.json',Path(run)/f'seed_{seed}'/'smoke'/'cache_checks.json')
            if not checks['passed']:
                raise RuntimeError('Smoke cache correctness failed. Fix implementation before training.')
            smoke_summary=read_json(Path(run)/f'seed_{seed}'/'smoke'/'training_summary.json')
            if smoke_summary['final_loss'] is not None and smoke_summary['final_loss']>=smoke_summary['initial_loss']:
                raise RuntimeError('Smoke training loss did not decrease. Investigate before the formal run.')
        elif stage=='train':
            for seed in seeds:
                train_transformer(cfg,run,seed,device,resume=args.resume)
        elif stage=='validate':
            failed=[]
            for seed in seeds:
                model=load_teacher(run,seed,device)
                cache=cache_correctness(cfg,run,seed,model,device)
                quality=teacher_quality(cfg,run,seed,model,device)
                if not cache['passed']:
                    raise RuntimeError(f'Seed {seed}: cache implementation checks failed.')
                if not quality['passed']:
                    failed.append(seed)
            if failed and cfg['profile']=='formal' and not args.diagnostic_continue:
                raise RuntimeError(f'Teacher learning gate failed for {failed}. Results saved; do not interpret the state hypothesis. Start a separately declared revised run or use --diagnostic-continue for explicitly nonconfirmatory debugging.')
        elif stage=='collect':
            require_learning(cfg,run,seeds,args)
            for seed in seeds:
                model=load_teacher(run,seed,device)
                for split in ('fit','analysis_val'):
                    collect_bank(cfg,run,seed,model,device,split)
        elif stage in ('preprocess','fit'):
            require_learning(cfg,run,seeds,args)
            for seed in seeds:
                bankroot=Path(run)/f'seed_{seed}'/'banks'
                fit_bank,val_bank=load_bank(bankroot/'fit'),load_bank(bankroot/'analysis_val')
                fit_preprocessors(cfg,run,seed,fit_bank,val_bank)
                if stage=='fit':
                    fit_all(cfg,run,seed,fit_bank,val_bank,device,resume=args.resume)
        elif stage=='evaluate':
            require_learning(cfg,run,seeds,args)
            for seed in seeds:
                model=load_teacher(run,seed,device)
                evaluate_seed(cfg,run,seed,model,device)
        elif stage=='plot':
            make_plots(cfg,run)
        elif stage=='report':
            report=build_report(cfg,run)
            print(json.dumps({k:v for k,v in report.items() if k!='seeds'},indent=2))
        mark(run,stage,'complete',seeds=seeds)
    except BaseException as exc:
        error=f'{type(exc).__name__}: {exc}'
        mark(run,stage,'failed',error=error)
        append_jsonl(Path(run)/'reports'/'failures.jsonl',dict(stage=stage,error=error,
                     utc=datetime.now(timezone.utc).isoformat(),seeds=seeds))
        (Path(run)/'reports'/'last_failure.md').write_text(
            f'# Experiment stopped at {stage}\n\n{error}\n\n'
            'Completed artifacts and measured failures are retained. Inspect manifest.json and the relevant '
            'teacher/quality.json, cache_checks.json, or fitting logs before interpreting the hypothesis.\n'
            'Resume with the same settings, or declare a separate exploratory revision; do not turn a failed gate into a pass.\n')
        raise


def require_learning(cfg,run,seeds,args):
    if cfg['profile']!='formal' or args.diagnostic_continue:
        return
    for seed in seeds:
        root=Path(run)/f'seed_{seed}'/'teacher'
        if not (root/'quality.json').exists() or not read_json(root/'quality.json')['passed']:
            raise RuntimeError('Run validate and pass teacher learning checks before fitting/evaluating.')
        if not read_json(root/'cache_checks.json')['passed']:
            raise RuntimeError('Cache correctness is a prerequisite.')


def main(argv=None):
    parser=argparse.ArgumentParser(description='PDF-faithful predictive-memory/density pilot')
    parser.add_argument('stage',choices=['doctor','init','verify','smoke','train','validate','collect','preprocess','fit','evaluate','plot','report','run'])
    parser.add_argument('--config',default='configs/pilot.yaml')
    parser.add_argument('--run-dir',default=None)
    parser.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seeds',type=int,nargs='+',default=None)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--diagnostic-continue',action='store_true',help='Bypass learning-only gates for debugging; permanently disables a formal green signal.')
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    set_seed(0)
    cfg=load_config(args.config)
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; run doctor in the GPU virtual environment or explicitly choose --device cpu.')
    if args.stage=='doctor':
        doctor(cfg,device)
        return
    if args.run_dir is None:
        if args.stage not in ('init','run'):
            parser.error('--run-dir is required for this stage.')
        args.run_dir=str(Path('runs')/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'__'+cfg['profile']+'__'+fingerprint(cfg)[:8]))
    run=Path(args.run_dir).resolve()
    initialize(cfg,run,device)
    if args.diagnostic_continue:
        obj=read_json(run/'manifest.json');obj['diagnostic_continue']=True;write_json(run/'manifest.json',obj)
    progress(f'Run directory: {run}')
    seeds=args.seeds or cfg['seeds']
    if not set(seeds)<=set(cfg['seeds']):
        parser.error('--seeds must be a subset of the configured teacher seeds.')
    if args.stage=='init':
        return
    stages=['verify','smoke','train','validate','collect','fit','evaluate','plot','report'] if args.stage=='run' else [args.stage]
    for stage in stages:
        record=read_json(run/'manifest.json')['stage_status'].get(stage,{})
        if args.resume and record.get('status')=='complete' and set(seeds)<=set(record.get('seeds',[])):
            progress(f'Skipping completed stage {stage}')
            continue
        execute_stage(stage,cfg,run,seeds,device,args)


if __name__=='__main__':
    main()
