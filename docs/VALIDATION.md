# Delivery validation — 22 October 2026

These are **software implementation checks**, not results from the proposed three-seed GPU study. They were run on Linux/Python 3.12.14 with no CUDA device exposed. PyTorch was `2.14.1+cu130`; numerical dependency versions are listed in `constraints-tested.txt`.

## Completed checks

- **26 tests passed**, covering exact generator probabilities, grouped splits, censoring, masks/causality, parameter budgets, cached replay, branch-safe swaps, no future-feature leakage, differentiability, valid instrument probabilities, operator completeness, repeated-eigenvalue gradients, absorbing joint trees, diagonal oracle realization, PCA without whitening, clean-prefix deduplication, balanced losses, autonomous updates, the distinguishability bound, JSON missing values, sealed selection, relocated checkpoints, formal-config protection, and classical training-time budgets.
- The **complete tiny profile** ran from generation through all three source conditions and all four state families, cache interventions, saved distributions, plots and reports. Its output was correctly `IMPLEMENTATION_ONLY`, with both scientific pass flags false.
- `scripts/validate_run.py` checked full-probability normalization, CSV/NPZ row mappings, joint-tree sizes and errors, the geometry bound, and the prohibition on a tiny-run green signal.
- Pair inspection exported actual fitted matrix real/imaginary entries and full future-comparison numbers. Restarting the completed pipeline with `--resume` skipped completed stages without retraining.
- `doctor --device cpu` passed the tensor, transformer and complex-gradient checks. Both shell wrappers passed Bash syntax validation. A distribution wheel was built from `pyproject.toml`.
- Plots from the tiny run were visually checked. These were poorly trained implementation-test outputs, not successful experimental curves. They are not bundled as scientific results.

## The proposal's 300-step correctness trial

This used the full prescribed architecture, seed 17, batch 32, and 300 optimizer steps on **CPU**, independently of the tiny run.

| Measurement | Observed |
|---|---:|
| Parameters | 803,592 |
| Initial training CE | 2.140043 |
| Final training CE | 0.823156 |
| Best validation CE | 0.852606 |
| Actual unmasked training targets | 276,348 |
| Training wall time in this environment | 40.50 seconds |
| Maximum full/cached logit difference, 32 smoke prefixes | 9.54e-7 |
| Maximum full/cached distribution TV | 3.46e-7 |
| Self-swap / full-donor replay TV | 0 / 0 |
| Residual-only forced-token control TV | 0 |

The smoke replay checks use 32 prefixes. The **formal** 1,024-prefix check is still required on the GPU with each trained teacher. Smoke loss reduction is not a teacher-learning gate pass and its checkpoint is never used for the formal study.

A small machine-readable record is in `docs/validation/software_checks.json`. Development runs and their weights are kept outside the deliverable; your machine will create its own clean runs.

## Still to run on your machine

The A800/5090 CUDA installation checks, GPU smoke trial, all three 6,000-step teachers, 108 analysis fits at full budgets, full held-out evaluations, GPU time/memory measurements, and scientific success gates have **not** been executed here. No novelty verification, quantum advantage, publication readiness, or positive density result is claimed.

Use the README's checks first, then `bash scripts/run_pilot.sh runs/pilot-v1`. Follow the measured gates even if the outcome is negative.
