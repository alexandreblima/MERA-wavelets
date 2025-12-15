#!/usr/bin/env bash
set -euo pipefail

TRACE=dataset/202301131400_bytes_1ms.csv
WINDOW=1024
STEP=1024
L=5
SEED=12345
OUT_ROOT="results/202301131400_bytes_1ms/grid_search"
mkdir -p "$OUT_ROOT"

run_job() {
  local label="$1"      # e.g., low, high
  local retains="$2"    # comma-separated retains
  local mse="$3"        # mse weight
  local suffix="$4"     # extra label for output dir

  local out_dir="$OUT_ROOT/${label}_mse_${suffix}"
  mkdir -p "$out_dir"
  scripts/run_mera_mawi.sh \
    --data="$TRACE" \
    --window-size="$WINDOW" --step="$STEP" \
    --num-windows=0 --start-window=1 \
    --mera-L="$L" --mera-chi=2 --mera-chimid=2 \
    --retains="$retains" \
    --preset=paper --seed="$SEED" \
    --warm-start-haar --hurst-preservation \
    --mse-weight="$mse" \
    --output="$out_dir/train.csv"
}

run_job "low" "0.01,0.02,0.05,0.1" 0.02 "002"
run_job "low" "0.01,0.02,0.05,0.1" 0.05 "005"
run_job "low" "0.01,0.02,0.05,0.1" 0.1  "010"
run_job "high" "0.2,0.4,0.8" 0.02 "002"
run_job "high" "0.2,0.4,0.8" 0.05 "005"
run_job "high" "0.2,0.4,0.8" 0.1  "010"
