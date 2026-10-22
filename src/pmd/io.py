from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, torch.Tensor):
        return jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(jsonable(row), allow_nan=False) + '\n')


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    keys = sorted({k for r in rows for k in r})
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows([{k: jsonable(v) for k, v in r.items()} for r in rows])


def save_torch(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_torch(path, device='cpu'):
    # Only load this repository's own local checkpoints, never untrusted pickle files.
    return torch.load(path, map_location=device, weights_only=False)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Explicit attention and highest precision for the FP32 parity checks.
    torch.set_float32_matmul_precision('highest')
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def environment(device='cpu'):
    result = {'python': sys.version, 'platform': platform.platform(),
              'torch': torch.__version__, 'torch_cuda_runtime': torch.version.cuda,
              'cuda_available': torch.cuda.is_available(), 'device': str(device)}
    if torch.cuda.is_available():
        result['gpus'] = [dict(index=i, name=torch.cuda.get_device_name(i),
                              total_bytes=torch.cuda.get_device_properties(i).total_memory)
                          for i in range(torch.cuda.device_count())]
    for label, args in [('pip_freeze', [sys.executable, '-m', 'pip', 'freeze']),
                        ('git_commit', ['git', 'rev-parse', 'HEAD']),
                        ('git_status', ['git', 'status', '--short']),
                        ('nvidia_smi', ['nvidia-smi'])]:
        try:
            p = subprocess.run(args, capture_output=True, text=True, timeout=20)
            result[label] = p.stdout.strip() if p.returncode == 0 else p.stderr.strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            result[label] = str(exc)
    return result


def progress(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)
