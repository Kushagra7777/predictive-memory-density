# Saved results and how to inspect them

All paths below are relative to a run directory, such as `runs/pilot-v1`. Files are grouped by teacher, source, model family, and restart. JSON writes and checkpoints use temporary files followed by atomic replacement. Large feature banks are compressed NPZ shards; detailed evaluations are CSV plus NumPy probability arrays. Failed stages are recorded in `manifest.json` and do not silently become completed stages.

## Provenance and training

| File | Contents |
|---|---|
| `manifest.json` | Creation time, resolved-config hash, profile, stage states/devices, diagnostic bypass flag. |
| `resolved_config.yaml` | Fully merged settings, including inherited values. |
| `environment/*.json` | Python/PyTorch/runtime/platform, package freeze, Git commit/status, visible GPU inventory and `nvidia-smi`; recorded initially and per executed stage. |
| `generator/oracle_checks.json` | Exact worked-pair checks and rational TV values. |
| `generator/split_manifest.json` | Exact family hash specification and all split family IDs. |
| `final_test_plan.json` | Outcome-independent held-out families, delays, shared fillers, tree membership. |
| `seed_*/teacher/train.jsonl` | Optimizer step, CE, LR and actual unmasked target count. |
| `seed_*/teacher/validation.jsonl` | Teacher-validation loss and checkpoint selection history. |
| `seed_*/teacher/{best,last}.pt` | Model, optimizer, config, step, RNG and selection state. |
| `seed_*/teacher/training_summary.json` | Parameters, precision, wall time, first 100 step durations, peak VRAM, actual targets versus padded slots. |
| `seed_*/teacher/cache_checks.json` | FP32 full/cached logit and TV errors, self/full-donor and residual-only controls. |
| `seed_*/teacher/quality.json` | Waiting/query errors to the generator, impossible mass, reset behavior. |

`smoke/` uses the same formats but separate checkpoints, smaller batch/step count and no scientific interpretation. `teacher/` is always trained independently from scratch.

## Clean feature banks

Each `seed_*/banks/{fit,analysis_val}/manifest.json` identifies completed `shard_00000.npz`, etc. Shard keys:

| Key | Shape / meaning |
|---|---|
| `last_residual` | `[roots,5,128]` pre-final-normalization current residual. |
| `all_layers` | `[roots,5,512]` four current block outputs. |
| `cache_summary` | `[roots,5,3072]` clue/current/middle K/V summary, head-major coordinates. |
| `teacher` | `[roots,5,8]` complete clean teacher probabilities at available offsets. |
| `active` | `[roots,5]` valid nonterminal observation/target mask. |
| `supplied` | `[roots,4]` observed continuation token IDs; terminal padding ignored. |
| `stage`, `position` | `[roots,5]` deterministic observable side information. |
| `roots`, `root_length` | Padded root IDs and true root length. |
| `family`, `clue`, `reset`, `base_delay` | Grouping/control metadata, **not encoder inputs**. |
| `belief` | `[roots,5]` known oracle posterior for the separate diagnostic, **not surrogate training target**. |

Never interpret an inactive slot's numeric arrays as an observed state. Preprocessing and fitting starts deduplicate complete valid token prefixes, not activation values.

`preprocessing/SOURCE.pkl` stores its fit-only normalization/PCA; adjacent JSON records shape, variance, storage and fit cost. `diagnostics/ridge_SOURCE_MODE.{pkl,json}` stores the native/processed posterior diagnostic and validation-selected penalty. Final diagnostic errors are in `diagnostics/held_out_posterior.csv` and include stage labels.

## Fitted state models

`seed_*/surrogates/SOURCE/FAMILY/restart_SEED/` contains training/validation JSONL, `best.pt`, `last.pt`, and `fit_summary.json`. The summary reports actual parameter counts, total updates, training time, wall time, GPU peak and classical time budget. Family values are `density2`, `classical4`, `vector3`, `classical2`; source values are `last_residual`, `all_layers`, `cache_summary`.

The family-level `selection.json` retains every restart's score and the validation-chosen restart. `selection_lock.json` hashes the teacher, preprocessing, selection files, and chosen surrogate checkpoints before final evaluation. Modified selected files are rejected during evaluation. Checkpoint paths can be relocated with a run; loading falls back to the run-relative model location.

## Evaluation CSVs

All are under `seed_*/evaluation/`.

| File | Unit and important columns |
|---|---|
| `pairs.csv` | One family/reset/cohort pair: current teacher TV, matched flag, shared-query gap, reset-query TV, teacher query error. |
| `ordinary.csv` | One source/model/root: `root_tv`, clue, cohort, delay, reset, matched, probability-array row. Both controlled ordinary and long roots are present. |
| `joints.csv` | One source/model/root/horizon: complete future `joint_tv`, normalization error, matched flag, array row. Horizons 1/2/3; all 8^k strings. |
| `effects.csv` | One source/model/family/recipient clue/reset/cohort/arm/continuation: actual and predicted signed effects, MAE, sign, donor-directed effect, P1, conditional future TV, actual/predicted log-ratio and distribution-TV changes, impossible mass, state access, array row. |
| `updates.csv` | Initial encoding followed by saved `supplied_tokens`, never refreshed: offsets 1/4/8, teacher output TV, distance to fresh clean encoding, stage and position. Stopped endpoints are excluded. |
| `strings.csv` | Actual and predicted probability mass of each specified supported continuation; selected strings, **not** a complete long joint. |
| `timeline.csv` | Representative matched paired histories before/after `ababab?`, actual/predicted P1 and full-distribution error at each step. |
| `geometry.csv` | Paired density trace distance, teacher current and three-token future TV, approximation-error upper bound, numerical bound violation. |
| `geometry_diagnostics.json` | Within source/cohort/delay/reset correlation and maximum bound violation; null correlation when not estimable. |
| `interactions.csv` | f11−f10−f01+f00 for answer probability. Same actual teacher effects repeat across model/source rows to retain common case identifiers. |
| `calibration.csv` | Predicted-P1 bins with counts, mean predicted/actual teacher P1, mean absolute error. This is agreement with a frozen teacher, not outcome-frequency calibration of the data generator. |
| `natural.csv` | All observed nonterminal post-clue tokens from naturally sampled test episodes; surrogate/teacher TV and teacher/oracle TV. Separate from controlled delays. |
| `benchmark.csv` | Batches 1/256; encoder, update, readout and CPU preprocessing seconds/call; state/storage/extra working-VRAM fields where applicable. |
| `audit_cache_pair.pt` | Small clean matched pair: histories, per-layer K/V, masks and probabilities; not an all-example raw-cache dump. |
| `complete.json` | Completion marker, record counts and no-refresh/no-intervention-fit flags. |

Read cache arm as a **two-character string**: `00` baseline, `10` earlier swap, `01` current swap, `11` full donor. Pandas may otherwise parse `01` as the integer 1. `donor_directed_effect` reverses the sign for recipient clue 1. `future_tv` in effects is the next-token distribution error **after conditioning on the supplied continuation**, not a whole-joint distance.

## Full probability arrays

`ordinary_distributions.npz` and `effects_distributions.npz` contain `teacher` and `surrogate` arrays shaped `[CSV_rows,8]`, plus `vocabulary`. Row order is exactly the CSV's `distribution_row` field.

```python
from pathlib import Path
import numpy as np
import pandas as pd

d = Path('runs/pilot-v1/seed_17/evaluation')
rows = pd.read_csv(d/'effects.csv', dtype={'arm': str})
data = np.load(d/'effects_distributions.npz')
case = rows.query("arm == '10' and continuation == '?' and source == 'cache_summary'").iloc[0]
i = int(case.distribution_row)
print(dict(zip(data['vocabulary'], data['teacher'][i])))
print(dict(zip(data['vocabulary'], data['surrogate'][i])))
```

`joint_distributions.npz` stores variable-size vectors concatenated into one-dimensional `teacher`/`surrogate` arrays. `offsets[i]:offsets[i+1]` selects row i of `joints.csv`; its length is exactly `8**horizon`. Branches follow lexicographic token-ID order, first token outermost.

```python
rows = pd.read_csv(d/'joints.csv')
data = np.load(d/'joint_distributions.npz')
row = rows.query('horizon == 3').iloc[0]
i = int(row.distribution_row)
lo, hi = data['offsets'][i:i+2]
p = data['teacher'][lo:hi].reshape(8,8,8)
print('P(? then 1 then $) =', p[5,2,7])
```

`density_trajectories.npz` holds bounded real `coordinates` and labels `[family,clue,reset,offset,stage,position]` for a small visualization subset; full evaluation metrics do not depend on this plotting cap. These coordinates reconstruct matrices using `rho_from_bloch`. Their axes are not aligned across teachers. `inspect_pair.py` exports specific matrices as separate real/imaginary arrays in organized JSON.

## Reports, failures, and plots

`seed_*/reports/gates.json` contains every measured predefined gate and the exact comparison. PASS/FAIL/INSUFFICIENT_DATA are distinct. `endpoint_summary.csv` reports all/retained-pair means, medians, p95 and whole-family 95% intervals for the primary cache-source endpoints and each model. Raw source comparisons remain available without filtering.

The combined `reports/decision.json`, `summary.md`, and `gates.csv` separate a causal-memory pilot from the stronger density-specific result. Three successful teacher seeds are required. Exploratory/tiny/debug results never establish the formal signal. If a formal learning gate stops the pipeline early, its measured quality/cache files remain available; run `report` to record that there is no green signal without fabricating missing downstream results.

Any interrupted/failed stage also writes `reports/last_failure.md` and appends `reports/failures.jsonl` automatically. A later successful resume preserves the failure history as provenance; consult current manifest/report status rather than interpreting an old failure file as the latest run decision.

PNG/SVG figures are generated from measured CSVs. Interactive Bloch HTML is self-contained. Error bars in the across-seed comparison are standard deviations of the measured teacher-seed means; they are not a population significance claim based on three seeds.

To back up a run independently of Git, copy its entire directory including `resolved_config.yaml`, provenance, model selections, and probability arrays. Keep the code commit and dependency freeze with it. Results are intentionally ignored by Git; they are not automatically pushed or deleted.
