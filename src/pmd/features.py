from __future__ import annotations

from pathlib import Path
import time
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from .banks import unique_state_indexes
from .io import read_json,write_json


class Preprocessor:
    def __init__(self):
        self.mean = self.scale = self.components = None

    def fit(self,x,width=128):
        self.mean = x.mean(0,dtype=np.float64).astype(np.float32)
        self.scale = np.maximum(x.std(0,dtype=np.float64),1e-6).astype(np.float32)
        standardized = (x-self.mean)/self.scale
        if x.shape[1]==width:
            self.components = None
        else:
            if min(x.shape)<width:
                raise ValueError(f'PCA requires at least {width} distinct fitting states.')
            pca = PCA(n_components=width,svd_solver='randomized',whiten=False,random_state=17)
            pca.fit(standardized)
            self.components = pca.components_.astype(np.float32)
            self.pca_mean = pca.mean_.astype(np.float32)
            self.explained_variance = float(pca.explained_variance_ratio_.sum())
        return self

    def transform(self,x):
        x = (x-self.mean)/self.scale
        if self.components is not None:
            x = (x-self.pca_mean) @ self.components.T
        return x.astype(np.float32)

    def save(self,path):
        joblib.dump(self,path)


def fit_preprocessors(cfg,run,seed,fit_bank,val_bank):
    directory = Path(run)/f'seed_{seed}'/'preprocessing'
    directory.mkdir(parents=True,exist_ok=True)
    indexes = unique_state_indexes(fit_bank)
    for source in cfg['analysis']['sources']:
        path = directory/f'{source}.pkl'
        if path.exists():
            continue
        tick=time.perf_counter()
        x = fit_bank[source][indexes]
        prep = Preprocessor().fit(x,cfg['analysis']['pca_width'])
        prep.save(path)
        write_json(directory/f'{source}.json',dict(source=source,fitted_split='fit',unique_fitting_states=len(x),
                   input_width=x.shape[1],output_width=128,whiten=False,
                   pca_explained_variance=getattr(prep,'explained_variance',None),
                   storage_bytes=path.stat().st_size,scale_floor=1e-6,
                   fit_seconds=time.perf_counter()-tick))
        del x
    fit_diagnostics(cfg,run,seed,fit_bank,val_bank)


def fit_diagnostics(cfg,run,seed,fit_bank,val_bank):
    directory = Path(run)/f'seed_{seed}'/'diagnostics'
    directory.mkdir(parents=True,exist_ok=True)
    fit_ix,val_ix=unique_state_indexes(fit_bank),unique_state_indexes(val_bank)
    for source in cfg['analysis']['sources']:
        for mode in ('native','processed'):
            path = directory/f'ridge_{source}_{mode}.pkl'
            if path.exists():
                continue
            prep = joblib.load(Path(run)/f'seed_{seed}'/'preprocessing'/f'{source}.pkl')
            x,xv = fit_bank[source][fit_ix],val_bank[source][val_ix]
            if mode=='processed':
                x,xv = prep.transform(x),prep.transform(xv)
            else:
                x,xv = (x-prep.mean)/prep.scale,(xv-prep.mean)/prep.scale
            target,validation = fit_bank['belief'][fit_ix],val_bank['belief'][val_ix]
            best = None
            for alpha in cfg['analysis']['ridge_alphas']:
                reg = Ridge(alpha=max(alpha,1e-12),solver='lsqr',tol=1e-5).fit(x,target)
                mse = float(np.square(reg.predict(xv)-validation).mean())
                if best is None or mse<best[0]:
                    best = mse,reg,alpha
            joblib.dump(best[1],path)
            write_json(path.with_suffix('.json'),dict(source=source,mode=mode,alpha=best[2],
                       validation_mse=best[0],target='oracle posterior diagnostic only',fitted_split='fit',
                       distinct_fit_states=len(target),distinct_validation_states=len(validation)))


def load_preprocessor(run,seed,source):
    return joblib.load(Path(run)/f'seed_{seed}'/'preprocessing'/f'{source}.pkl')
