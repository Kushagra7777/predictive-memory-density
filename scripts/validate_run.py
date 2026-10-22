#!/usr/bin/env python3
"""Validate saved distributions and row mappings in a completed run."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from pmd.io import read_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True)
    args=parser.parse_args();run=Path(args.run_dir)
    result=read_json(run/'reports'/'decision.json')
    for seed in result['evaluated_seeds']:
        d=run/f'seed_{seed}'/'evaluation'
        for name in ['ordinary','effects']:
            rows=pd.read_csv(d/f'{name}.csv')
            data=np.load(d/f'{name}_distributions.npz')
            assert np.array_equal(rows.distribution_row,np.arange(len(rows)))
            for label in ('teacher','surrogate'):
                a=data[label]
                assert a.shape==(len(rows),8) and np.isfinite(a).all()
                assert a.min()>=-1e-6 and np.allclose(a.sum(-1),1,atol=3e-6)
            values=np.abs(data['teacher']-data['surrogate']).sum(-1)/2
            expected=rows.root_tv if name=='ordinary' else rows.future_tv
            assert np.allclose(values,expected,atol=1e-6)
        rows=pd.read_csv(d/'joints.csv');data=np.load(d/'joint_distributions.npz')
        assert len(data['offsets'])==len(rows)+1
        for row in rows.itertuples():
            lo,hi=data['offsets'][row.distribution_row:row.distribution_row+2]
            assert hi-lo==8**row.horizon
            p,q=data['teacher'][lo:hi],data['surrogate'][lo:hi]
            assert np.isclose(p.sum(),1,atol=3e-6) and np.isclose(q.sum(),1,atol=3e-6)
            assert np.isclose(np.abs(p-q).sum()/2,row.joint_tv,atol=1e-6)
        geom=pd.read_csv(d/'geometry.csv')
        assert geom.bound_violation.max()<3e-6
        assert read_json(d/'complete.json')['teacher_refreshed_surrogate'] is False
    if result['profile']!='formal':
        assert result['status']=='IMPLEMENTATION_ONLY'
        assert not result['density_specific_pass'] and not result['memory_pilot_pass']
    print(f"Saved distributions, row mappings, and claim guard passed for {result['evaluated_seeds']}; status={result['status']}")


if __name__=='__main__':
    main()
