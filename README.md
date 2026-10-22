# Same prediction now, different memory later

A complete implementation of **chatgpt-pro-research-methodology.pdf**, dated 21 October 2026. Two histories can give the same prediction now while retaining different information that matters after the same later token. This project trains a small transformer, fits compact descriptions of its internal state, and tests their predictions under held-out changes to the transformer's attention memory.

The candidate description is a **2-by-2 complex density matrix**, stored as three real numbers. It is compared with the proposal's four-state classical probabilistic model, three-number recurrent model, and smaller two-state diagnostic. This is ordinary numerical computation on a GPU. A useful result would concern a predictive modeling constraint; it would not establish physical quantum computation inside a transformer.

**No pretrained model or dataset download is needed.** The exact eight-token generator and the 803,592-parameter transformer are implemented here. The transformer is trained from random initialization. No account, API key, Hugging Face checkpoint, tokenizer, quantum computer, Qiskit, or PennyLane is needed.

The source proposal is included in [docs/proposal/chatgpt-pro-research-methodology.pdf](docs/proposal/chatgpt-pro-research-methodology.pdf). Read [protocol coverage](docs/PROTOCOL_COVERAGE.md) for the section-by-section implementation map and [implementation decisions](docs/IMPLEMENTATION_DECISIONS.md) for choices the proposal leaves unspecified.

## 1. Clone and install on the GPU machine

Use **Python 3.12** in a fresh environment. The tested dependency versions are in `constraints-tested.txt`; `requirements.txt` lists the supported dependency ranges. Python 3.10+ is allowed by the package, but the constrained versions were checked with 3.12.

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd predictive-memory-density
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "torch==2.14.1"
python -m pip install -r requirements-dev.txt -c constraints-tested.txt
python -m pip install -e . --no-deps
```

The GPU installation above uses the published PyTorch wheel. The build used for the CPU checks in this delivery was `2.14.1+cu130`. Installation availability varies by OS/Python; if the chosen wheel is unavailable, use the official [PyTorch installation selector](https://pytorch.org/get-started/locally/) or [version instructions](https://pytorch.org/get-started/previous-versions/), then run `doctor` below. Install a CUDA-enabled wheel on Linux; do not use a CPU-only wheel for GPU experiments. The project uses wheel-provided libraries and needs no custom CUDA compilation. Do not modify existing ComfyUI or action-segmentation environments.

NVIDIA documents CUDA 13.x minor-version compatibility with driver branches 580 and newer: [compatibility guide](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html). Your supplied 590.48.01 driver is within that range. The `CUDA Version` displayed by `nvidia-smi` is not the Python package version. Actual compatibility must pass the tensor, attention, and complex-gradient checks below.

### Select the A800

Your supplied snapshot shows physical GPU 1, the **A800 40GB**, mostly free. The pilot uses **one GPU**. Select it in the same terminal before running commands:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
python -m pmd doctor --device cuda:0
```

After masking, physical GPU 1 is logical **`cuda:0`** inside Python. Do not pass `cuda:1` in that terminal. Check the GPU name printed by `doctor`. GPU numbering can change after a reboot: inspect `nvidia-smi` first. You can select the 5090 instead with `CUDA_VISIBLE_DEVICES=0` when it is available; this delivery has not been tested on either of your GPUs.

The proposal assumes at least **32 GB host RAM**, one 24–32 GB GPU, and local disk. Your A800's 40 GB exceeds that assumed GPU capacity. Allow roughly **20 GB free disk initially**, then inspect actual usage; this is a working allowance, not a measured full-pilot requirement. Feature banks and detailed per-case results dominate storage. No distributed training or combination of the two GPUs is required.

## 2. Run the checks first

Run from the repository root with the environment active:

```bash
python -m pytest -q
python -m pmd verify --config configs/pilot.yaml --run-dir runs/checks --device cuda:0
bash scripts/run_smoke.sh runs/smoke-v1
```

The smoke trial uses the specified 4-layer architecture, seed 17, batch 32, and 300 optimizer steps. It checks loss decrease and cached/full-forward agreement; its weights are saved under `smoke/` and never used as the formal teacher. Examine `runs/smoke-v1/seed_17/smoke/training_summary.json` and `cache_checks.json`.

For a fast **complete software check**, including every model family, interventions, plots, and reporting:

```bash
python -m pmd run --config configs/tiny.yaml --run-dir runs/tiny-check --device cpu --threads 2
python scripts/validate_run.py --run-dir runs/tiny-check
```

The tiny profile deliberately uses almost no training. Its only possible decision label is `IMPLEMENTATION_ONLY`. Bad predictions in this run are expected. It cannot produce a scientific green signal.

## 3. Run the full proposal

```bash
bash scripts/run_pilot.sh runs/pilot-v1
```

This wrapper runs the following command and keeps a console log:

```bash
python -u -m pmd run --config configs/pilot.yaml --run-dir runs/pilot-v1 --device cuda:0 --resume
```

The complete pipeline performs these stages in order:

1. Verify the exact generator, reset identities, and worked-pair probabilities.
2. Run the separate 300-step correctness trial.
3. Train teachers from scratch for seeds **17, 29, 43**, each for **6,000 steps** at batch **128**; select each checkpoint by transformer-validation loss.
4. Check 1,024 cached/full prefixes and the transformer learning gates. **Stop before analysis if a formal teacher fails.**
5. Collect 20,000 distinct clean fitting roots and 4,000 validation roots per teacher, with the specified continuation mixture.
6. Fit/freeze standardization, unwhitened PCA, and native/compressed posterior diagnostics. Fit **3 sources × 4 model families × 3 restarts × 3 teachers = 108 surrogate fits**. Matched classical fits receive at least 3,000 updates and the density model's measured training-time budget.
7. Seal selected checkpoints. Evaluate 1,024 ordinary and 512 further long-delay families, complete short future trees, autonomous updates, both cache-swap directions, all four arms, and reset controls.
8. Save distributions, per-case errors, whole-family bootstrap intervals, plots, and the final gate report.

The PDF estimates **4–14 GPU-hours** for the pilot. That is an estimate from the proposal, not a measured promise. The first 100 teacher-step timings, representative analysis fit times, peak GPU memory, preprocessing cost, and inference timings are saved. Use your own measurements to judge the remaining runtime. Training may be fast for this small model; analysis and many controlled replays can dominate.

Use `tmux` or your existing job scheduler for a long remote session. Run only **one writer per run directory**. The code does not coordinate competing processes.

### Resume after interruption

Run the same wrapper and the same run path again:

```bash
bash scripts/run_pilot.sh runs/pilot-v1
```

Teacher and surrogate optimizers resume at their last validation checkpoint, including their data RNG state. Collection restarts an incomplete bank deterministically; completed banks are reused. An interrupted evaluation restarts that seed's evaluation, without refitting. Completed stages are skipped by `--resume`. Keep the code revision and dependencies fixed while resuming. A different resolved configuration is rejected in an existing run directory.

Checkpoints are local `torch`/`joblib` files intended for this run; load only your own trusted experiment artifacts.

### Run stages or teachers separately

```bash
python -m pmd train --run-dir runs/pilot-v1 --device cuda:0 --seeds 17 --resume
python -m pmd validate --run-dir runs/pilot-v1 --device cuda:0 --seeds 17
python -m pmd collect --run-dir runs/pilot-v1 --device cuda:0 --seeds 17
python -m pmd fit --run-dir runs/pilot-v1 --device cuda:0 --seeds 17 --resume
python -m pmd evaluate --run-dir runs/pilot-v1 --device cuda:0 --seeds 17
python -m pmd plot --run-dir runs/pilot-v1 --device cpu
python -m pmd report --run-dir runs/pilot-v1 --device cpu
```

All commands default to `configs/pilot.yaml`; explicitly supply another config for a nonformal run. The `preprocess` stage can be run separately before `fit`. A single teacher cannot satisfy the three-seed decision rule.

## 4. Find the results

Everything is saved automatically under the chosen run directory; no results need to be copied by hand. Start with:

```text
runs/pilot-v1/
  manifest.json                       stage status, configuration hash, provenance
  resolved_config.yaml                complete settings, including inherited settings
  environment/                        Python, packages, Git state, GPU details per stage
  generator/                          exact checks and family split manifest
  final_test_plan.json                 sealed test families and waiting strings
  seed_17/                            repeated independently for 29 and 43
    smoke/                            separate correctness weights and logs
    teacher/                          best/last weights, losses, quality/cache checks
    banks/{fit,analysis_val}/          compressed feature shards and manifests
    preprocessing/                    frozen fit-only normalization and PCA
    diagnostics/                      native/compressed ridge models and test errors
    surrogates/SOURCE/FAMILY/
      restart_SEED/                   best/last weights, optimizer state, JSONL logs
      selection.json                  all restart scores and validation selection
    selection_lock.json               hashes of selected fitting artifacts
    evaluation/                       per-case CSVs, full distributions, audit cache
    reports/                          measured gates and endpoint summaries
  plots/                              PNG + SVG figures and interactive Bloch HTML
  reports/{summary.md,decision.json,gates.csv}
  logs/console.log                    created by the shell wrapper
  inspections/                        optional inspect_pair exports
```

Read `reports/summary.md` first, then `seed_*/reports/gates.json` for actual numbers, thresholds, and family intervals. `endpoint_summary.csv` compares all controlled pairs with the retained current-distribution-matched subset. The raw files retain every information source and case, including failures. See [RESULTS_FORMAT.md](docs/RESULTS_FORMAT.md) for exact fields and NumPy examples.

Expected decision labels:

| Label | Meaning |
|---|---|
| `IMPLEMENTATION_ONLY` | Tiny or exploratory profile; no scientific success claim. |
| `NO_GREEN_SIGNAL` | The formal prerequisites or relevant criteria have not all passed. Inspect individual gates. |
| `MEMORY_PILOT_PASS` | All three teachers pass prerequisite checks and the specified causal memory effect replicates. This alone does not establish a useful density model. |
| `DENSITY_PILOT_PASS` | All absolute, update, held-out intervention, long-delay, and matched/time-budget classical comparisons pass for all three teachers. |

Every report sets `novelty_verified: false`. Positive results support a further study, not publication readiness, uniqueness, or an irreducible quantum advantage. Existing related work is documented in [RELATED_WORK.md](docs/RELATED_WORK.md).

### Look at one pair and its actual matrices

```bash
python scripts/inspect_pair.py --run-dir runs/pilot-v1 --seed 17 --device cuda:0
python scripts/inspect_pair.py --run-dir runs/pilot-v1 --seed 17 --reset --device cuda:0
python scripts/validate_run.py --run-dir runs/pilot-v1
```

The inspection command prints the two token histories, real and imaginary matrix entries, trace distance, current transformer probabilities, shared-query predictions, and complete 1/2/3-token future distances. It saves the same information as organized JSON. Add `--family NUMBER` to choose a family from `final_test_plan.json`; add `--source last_residual` or `all_layers` to inspect another source. It does not train or alter the selected models.

Open `plots/seed_17_bloch_ball.html` in a browser for a rotatable three-dimensional view. Other figures show prediction timelines, memory swaps, predicted/actual effects, calibration, autonomous updates, long-delay errors, interactions, posterior diagnostics, training, and latency. Bloch axes are coordinates, not named semantic variables; across-teacher axes are not aligned.

## 5. Keep the experiment honest

`configs/pilot.yaml` is the frozen proposal configuration. A formal run is checked against `docs/protocol_fingerprint.json`. To revise training or hypotheses after a failure, copy the config, set `profile: exploratory`, choose a **new** run directory, and record why you changed it. Do not relax a threshold after seeing final-test outcomes and describe that as the original experiment.

`--diagnostic-continue` allows explicitly nonconfirmatory debugging after a teacher-learning failure. Its use is permanently recorded and disables a formal green signal for that run. Cache-correctness failure is never bypassed.

The main intervention changes the cache **after** current predictions have already been computed, then supplies an identical continuation before scoring. It never reprocesses the old prefix. Residual-only encoders receive unchanged recipient vectors; failure there can reflect lack of access to the edit. The fair density-versus-classical intervention comparison is **within the cache-summary source**.

A full failed run is useful evidence. It can show that the teacher did not learn the task, that the selected compression loses information, that autonomous updates fail, or that ordinary classical models work as well. A density matrix being mathematically valid is only a starting check.

## Code map

| Path | Responsibility |
|---|---|
| `src/pmd/generator.py`, `reference.py` | Exact simulator, posterior oracle, splits, diagonal matrix reference. |
| `transformer.py`, `train.py` | Explicit causal attention, branch-safe KV cache, teacher training/checkpoints. |
| `banks.py`, `features.py` | Clean feature extraction, deduplication, frozen fit-only preprocessing and ridge diagnostics. |
| `surrogates.py`, `fit.py` | Density instrument, classical opponents, encoders, autonomous loss, time matching. |
| `evaluate.py`, `metrics.py` | Full trees, held-out edits, reset controls, distributions, distances, family bootstrap. |
| `report.py`, `plots.py`, `cli.py` | Predefined decisions, figures, orchestration and provenance. |
| `tests/`, `scripts/`, `configs/` | Invariant checks, inspection/run helpers, fixed and tiny profiles. |

This delivery's actual software checks are recorded in [VALIDATION.md](docs/VALIDATION.md). No GPU research result is bundled or claimed. Choose a public code license before advertising reuse terms; no license has been selected on your behalf.
