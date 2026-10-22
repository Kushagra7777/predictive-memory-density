from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml


def merge(base, update):
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    parent = cfg.pop("extends", None)
    if parent:
        cfg = merge(load_config(path.parent / parent), cfg)
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    assert cfg['model']['vocab_size'] == 8, 'The protocol uses exactly eight tokens.'
    assert cfg['model']['width'] % cfg['model']['heads'] == 0
    assert cfg['generator']['max_context'] == cfg['model']['context']
    assert sum(cfg['generator']['waiting_probs']) == 1.0
    assert cfg['analysis']['pca_width'] == cfg['model']['width'] == 128
    assert cfg['analysis']['rollout'] == 4
    assert cfg['generator']['nuisance_length'] == 16
    assert cfg['generator']['reliability'] == .8, 'Changing the task also requires changing its exact oracle.'
    assert cfg['generator']['waiting_probs'] == [.4,.4,.1,.1]
    assert cfg['model']['context'] == 64
    assert (cfg['model']['layers'],cfg['model']['heads'],cfg['model']['ff_width']) == (4,4,512)
    if cfg['profile'] == 'formal':
        assert cfg['seeds'] == [17, 29, 43]
        assert cfg['train']['steps'] == 6000
        assert cfg['analysis']['roots_fit'] == 20000


def fingerprint(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
