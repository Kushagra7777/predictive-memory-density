#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
run_dir="${1:-runs/pilot-v1}"
mkdir -p "$run_dir/logs"
python -u -m pmd run --config configs/pilot.yaml --run-dir "$run_dir" --device cuda:0 --resume \
  2>&1 | tee -a "$run_dir/logs/console.log"
