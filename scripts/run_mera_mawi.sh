#!/usr/bin/env bash
set -euo pipefail

# Generic runner for MERA training/eval on MAWI-like CSV series.
# Defaults tuned for 1D series (L=3, chi=2, chi_mid=2) and maximum available windows.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Ensure Julia uses the repo project by default
export JULIA_PROJECT="$ROOT_DIR"

print_usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options (long --key=value or toggles):
  --data=PATH                    CSV file (default: dataset/202301131400_bytes_1ms.csv)
  --window-size=N                Window length (default: 4096)
  --step=N                       Step between windows (default: 4096)
  --num-windows=N                How many windows to process (default: 0 = all available)
  --all-windows                  Shorthand: use all available windows (sets --num-windows=0)
  --start-window=N               1-based starting window index (default: 1)
  --retains=CSV                  Retain rates CSV (default: 0.01,0.02,0.05,0.1,0.2,0.4,0.8)
  --auto-data                    Auto-detect input CSV in dataset/ (pick newest)
  --data-pattern=GLOB            Glob pattern for auto-detect (default: dataset/*bytes_1ms.csv; fallback: dataset/*.csv)
  --mera-L=N                     Number of levels (default: 3)
  --mera-chi=N                   Chi (bond dim) (default: 2)
  --mera-chimid=N                Chi_mid (intermediate) (default: 2)
  --stage1-iters=N               Stage 1 iterations (default: 25)
  --stage1-lr=FLOAT              Stage 1 LR (default: 5e-3)
  --stage2-iters=N               Stage 2 iterations (default: 25)
  --stage2-lr=FLOAT              Stage 2 LR (default: 2.5e-3)
  --output=PATH                  Output CSV (default: results/debug_train_mawi_5.csv)
  --save-models                  Save per-window model checkpoints (wC/vC) in JLD2 next to output (models/)
  --models-dir=PATH              Custom directory to save models (defaults to <dirname(output)>/models)
  --warm-start-haar | --no-warm-start-haar   Toggle Haar init (default: ON)
  --train | --no-train           Toggle training (default: ON)
  --threads=N                    JULIA_NUM_THREADS (default: 1)
  --parallel-windows             Enable inter-window CPU parallelism (uses varMERA runner)
  --blas-threads=N               BLAS threads (default: 1 when parallel-windows; otherwise unchanged)
  --no-parallel-windows          Force sequential (clean runner)
  --seed=INT                     Fix Random.seed! for reproducibility (default: 12345 in runners)
  --preset=NAME                  Preset for iterations: fast|paper|quality (sets stage iters)
  --pin-threads                  Pin Julia threads to cores (JULIA_EXCLUSIVE=1) and align BLAS threads
  --dry-run                      Print resolved command and exit
  --date-stamp                   Append today's date (YYYY-MM-DD) to default output filename (only when --output is not provided)
  --param=NAME                   Parametrization: mera (default) | angle (minimal O(2))
  --early-stop                   Enable early stopping (relative tol + patience)
  --tol=FLOAT                    Relative tolerance (default: 1e-4)
  --patience=N                   Patience in iterations without improvement (default: 5)
  --min-iter=N                   Minimum iterations before checking early stop (default: 10)
  --sparsity-weight=FLOAT        Weight for L1 sparsity term (default: 1.0)
  --mse-weight=FLOAT             Weight for MSE reconstruction term (default: 0.0 = disabled)
  --post-strict                  Make post-run analysis errors fatal (default: non-fatal)
  --no-periodogram               Skip smoothed periodogram plots
  --no-aggregation               Skip multiscale aggregation plot
  --baseline-wavelets            Compute baseline DWT PSNRs (haar, db4, c3, s8, bior4.4)
  --no-baseline-wavelets         Skip baseline DWT PSNRs
  --baseline-waves=CSV           Comma-separated list of baselines (default: haar,db4,c3,s8,bior4.4)
  --hurst-preservation           Run ΔH vs retain analysis (compare_hurst_retains + plot)
  --no-hurst-preservation        Skip ΔH analysis (default)
  --hurst-retain-min=FLOAT       Min retain for ΔH sweep (default: 0.01)
  --hurst-retain-max=FLOAT       Max retain for ΔH sweep (default: 0.8)
  --hurst-retain-steps=N         Steps in retain sweep (default: 7)
  --hurst-num-windows=N          Windows to use for ΔH (default: 12)
  --hurst-iters1=N               Iterations stage 1 (ΔH inner learned) (default: 3)
  --hurst-iters2=N               Iterations stage 2 (ΔH inner learned) (default: 3)
  --hurst-init=haar|random       Init for ΔH learned run (default: haar)
  -h, --help                     Show this help

Examples:
  $(basename "$0") --num-windows=10 --output=results/run10.csv
  $(basename "$0") --retains=0.02,0.05 --threads=8 --no-train
EOF
}

# Mandatory wavelets for baseline comparisons (canonical names)
MANDATORY_BASELINE_WAVES=("haar" "db4" "c3" "s8" "bior4.4")

normalize_wavelet_token() {
  local token="${1,,}"
  token="${token// /}"
  token="${token//-}"
  case "$token" in
    "" ) echo "" ;;
    db1|haar|daubechies1) echo "haar" ;;
    db4|daubechies4) echo "db4" ;;
    c3|coif3|coiflet3|coif6) echo "c3" ;;
    s8|sym8|symmlet8|symlet8) echo "s8" ;;
    bior44|bior4.4|biorthogonal44|biorthogonal4.4|cdf97) echo "bior4.4" ;;
    *) echo "$token" ;;
  esac
}

canonical_display_wavelet() {
  case "$1" in
    haar) echo "haar" ;;
    db4) echo "db4" ;;
    c3) echo "c3" ;;
    s8) echo "s8" ;;
    bior4.4) echo "bior4.4" ;;
    *) echo "$1" ;;
  esac
}

# Defaults
DATA="dataset/202004081229_bytes_1ms.csv"
WINDOW_SIZE=1024
STEP=1024
# 0 means: use the maximum number of windows available in the dataset
NUM_WINDOWS=0
START_WINDOW=1
RETAINS="0.01,0.02,0.05,0.1,0.2,0.4,0.8"
MERA_L=5
MERA_CHI=2
MERA_CHIMID=2
STAGE1_ITERS=50
STAGE1_LR=5e-3
STAGE2_ITERS=50
STAGE2_LR=2.5e-3
OUTPUT=""  # if not provided, will be derived from --data and --mera-L
OUTPUT_SET=0
WARM_START=1
DO_TRAIN=1
# Threads: if not provided, will auto-detect later (nproc) for parallel mode
THREADS=""
# Default to parallel windows ON (proved faster/stable on CPU)
PARALLEL_WINDOWS=1
BLAS_THREADS=""
SEED=""
# Optional tuning
PRESET=""
PARAM="mera"
PIN_THREADS=0
DRY_RUN=0
DO_PERIODOGRAM=1
DO_AGG=1
DO_BASELINE_WAVES=1
BASELINE_WAVES="haar,db4,c3,s8,bior4.4"
DO_HURST=1
DATE_STAMP=0
SAVE_MODELS=0
MODELS_DIR=""
POST_STRICT=0
EARLY_STOP=1
TOL=1e-4
PATIENCE=5
MIN_ITER=10
SPARSITY_WEIGHT=1.0
MSE_WEIGHT=0.0
HURST_RETAIN_MIN=0.01
HURST_RETAIN_MAX=0.8
HURST_RETAIN_STEPS=7
HURST_NUM_WINDOWS=12
HURST_ITERS1=3
HURST_ITERS2=3
HURST_INIT="haar"

# Auto data detection options
AUTO_DATA=0
DATA_PATTERN=""

# Parse long options --key=value and toggles
for arg in "$@"; do
  case "$arg" in
    -h|--help) print_usage; exit 0 ;;
    --data=*) DATA="${arg#*=}" ;;
    --window-size=*) WINDOW_SIZE="${arg#*=}" ;;
    --step=*) STEP="${arg#*=}" ;;
    --num-windows=*) NUM_WINDOWS="${arg#*=}" ;;
    --all-windows) NUM_WINDOWS=0 ;;
    --start-window=*) START_WINDOW="${arg#*=}" ;;
    --retains=*) RETAINS="${arg#*=}" ;;
  --auto-data) AUTO_DATA=1 ;;
  --data-pattern=*) DATA_PATTERN="${arg#*=}" ;;
    --mera-L=*) MERA_L="${arg#*=}" ;;
    --mera-chi=*) MERA_CHI="${arg#*=}" ;;
    --mera-chimid=*) MERA_CHIMID="${arg#*=}" ;;
    --stage1-iters=*) STAGE1_ITERS="${arg#*=}" ;;
    --stage1-lr=*) STAGE1_LR="${arg#*=}" ;;
    --stage2-iters=*) STAGE2_ITERS="${arg#*=}" ;;
    --stage2-lr=*) STAGE2_LR="${arg#*=}" ;;
  --output=*) OUTPUT="${arg#*=}"; OUTPUT_SET=1 ;;
    --save-models) SAVE_MODELS=1 ;;
    --models-dir=*) MODELS_DIR="${arg#*=}" ;;
    --warm-start-haar) WARM_START=1 ;;
    --no-warm-start-haar) WARM_START=0 ;;
    # --train e --no-train ignorados, método variacional sempre ativado
    --train) : ;;
    --no-train) : ;;
    --threads=*) THREADS="${arg#*=}" ;;
    --parallel-windows) PARALLEL_WINDOWS=1 ;;
    --no-parallel-windows) PARALLEL_WINDOWS=0 ;;
    --blas-threads=*) BLAS_THREADS="${arg#*=}" ;;
    --seed=*) SEED="${arg#*=}" ;;
    --preset=*) PRESET="${arg#*=}" ;;
  --param=*) PARAM="${arg#*=}" ;;
    --pin-threads) PIN_THREADS=1 ;;
    --early-stop) EARLY_STOP=1 ;;
    --tol=*) TOL="${arg#*=}" ;;
    --patience=*) PATIENCE="${arg#*=}" ;;
    --min-iter=*) MIN_ITER="${arg#*=}" ;;
    --sparsity-weight=*) SPARSITY_WEIGHT="${arg#*=}" ;;
    --mse-weight=*) MSE_WEIGHT="${arg#*=}" ;;
    --post-strict) POST_STRICT=1 ;;
  --no-periodogram) DO_PERIODOGRAM=0 ;;
  --no-aggregation) DO_AGG=0 ;;
    --baseline-wavelets) DO_BASELINE_WAVES=1 ;;
    --no-baseline-wavelets) DO_BASELINE_WAVES=0 ;;
    --baseline-waves=*) BASELINE_WAVES="${arg#*=}" ;;
    --date-stamp) DATE_STAMP=1 ;;
    --hurst-preservation) DO_HURST=1 ;;
    --no-hurst-preservation) DO_HURST=0 ;;
    --hurst-retain-min=*) HURST_RETAIN_MIN="${arg#*=}" ;;
    --hurst-retain-max=*) HURST_RETAIN_MAX="${arg#*=}" ;;
    --hurst-retain-steps=*) HURST_RETAIN_STEPS="${arg#*=}" ;;
    --hurst-num-windows=*) HURST_NUM_WINDOWS="${arg#*=}" ;;
    --hurst-iters1=*) HURST_ITERS1="${arg#*=}" ;;
    --hurst-iters2=*) HURST_ITERS2="${arg#*=}" ;;
    --hurst-init=*) HURST_INIT="${arg#*=}" ;;
    --dry-run) DRY_RUN=1 ;;
    *) echo "Unknown option: $arg" >&2; print_usage; exit 2 ;;
  esac
done

# Ensure mandatory baseline wavelets are always present when baseline computation is enabled
if [[ "${DO_BASELINE_WAVES}" == "1" ]]; then
  IFS=',' read -ra _user_waves <<< "${BASELINE_WAVES}"
  declare -A _seen_waves=()
  final_waves=()
  for wave in "${_user_waves[@]}"; do
    wave="${wave//[[:space:]]/}"
    [[ -z "${wave}" ]] && continue
    canonical="$(normalize_wavelet_token "${wave}")"
    [[ -z "${canonical}" ]] && continue
    display="$(canonical_display_wavelet "${canonical}")"
    if [[ -z "${_seen_waves[$canonical]+x}" ]]; then
      final_waves+=("${display}")
      _seen_waves[$canonical]=1
    fi
  done
  for mandatory in "${MANDATORY_BASELINE_WAVES[@]}"; do
    canonical="$(normalize_wavelet_token "${mandatory}")"
    [[ -z "${canonical}" ]] && continue
    if [[ -z "${_seen_waves[$canonical]+x}" ]]; then
      display="$(canonical_display_wavelet "${canonical}")"
      final_waves+=("${display}")
      _seen_waves[$canonical]=1
    fi
  done
  if [[ ${#final_waves[@]} -eq 0 ]]; then
    final_waves=("${MANDATORY_BASELINE_WAVES[@]}")
  fi
  BASELINE_WAVES="$(IFS=','; echo "${final_waves[*]}")"
fi

# Diagnóstico global e Hurst CI (opcional)
TRACE_FILE="${DATA##*/}"
TRACE_BASE="${TRACE_FILE%.*}"
OUT_DIR="results/${TRACE_BASE}/L_${MERA_L}"
mkdir -p "${OUT_DIR}"

echo "[post] Diagnóstico global de multifractalidade do trace"
MULTIFRAC_CSV="${OUT_DIR}/multifractal_diagnosis.csv"
MULTIFRAC_PNG="${OUT_DIR}/multifractal_spectrum.png"
if ! julia --project=. scripts/analysis/diagnose_multifractality.jl \
  "${DATA}" \
  "${MULTIFRAC_CSV}" \
  "${MULTIFRAC_PNG}"; then
  echo "[post][warn] diagnose_multifractality failed (non-fatal)" >&2
  if [[ "${POST_STRICT}" == "1" ]]; then echo "[post][fatal] aborting due to --post-strict" >&2; exit 1; fi
fi

echo "[post] Estimando parâmetro de Hurst do trace original com IC"
HURST_RAW_CSV="${OUT_DIR}/hurst_raw_summary.csv"
if ! julia --project=. scripts/analysis/compute_hurst_ci.jl \
  --data "${DATA}" \
  --out "${HURST_RAW_CSV}"; then
  echo "[post][warn] compute_hurst_ci failed (non-fatal)" >&2
  if [[ "${POST_STRICT}" == "1" ]]; then echo "[post][fatal] aborting due to --post-strict" >&2; exit 1; fi
fi

# Optional: multifractal diagnosis & Hurst CI (moved to the end to avoid early warnings)
# echo "[post] Diagnóstico global de multifractalidade do trace"
# TRACE_FILE="${DATA##*/}"
# TRACE_BASE="${TRACE_FILE%.*}"
# OUT_DIR="results/${TRACE_BASE}/L_${MERA_L}"
# MULTIFRAC_CSV="${OUT_DIR}/multifractal_diagnosis.csv}"
# MULTIFRAC_PNG="${OUT_DIR}/multifractal_spectrum.png}"
# julia --project=. scripts/analysis/diagnose_multifractality.jl \
#   "${DATA}" \
#   "${MULTIFRAC_CSV}" \
#   "${MULTIFRAC_PNG}"
# echo "[post] Estimando parâmetro de Hurst do trace original com IC"
# HURST_RAW_CSV="${OUT_DIR}/hurst_raw_summary.csv"
# julia --project=. scripts/analysis/compute_hurst_ci.jl \
#   --data "${DATA}" \
#   --mera "${MERA_CSV}" \
#   --out "${HURST_RAW_CSV}"

# Auto-detect threads if not set/empty
if [[ -z "${THREADS}" || "${THREADS}" == "0" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    THREADS="$(nproc)"
  else
    THREADS="4"
  fi
fi

: "${JULIA_NUM_THREADS:=${THREADS}}"
export JULIA_NUM_THREADS

# Normalize JULIA_NUM_THREADS to a positive integer (handle values like 'auto')
if ! [[ "${JULIA_NUM_THREADS}" =~ ^[0-9]+$ ]] || [[ "${JULIA_NUM_THREADS}" -le 0 ]]; then
  if command -v nproc >/dev/null 2>&1; then
    JULIA_NUM_THREADS="$(nproc)"
  else
    JULIA_NUM_THREADS="${THREADS}"
  fi
  export JULIA_NUM_THREADS
fi

echo "[run] JULIA_NUM_THREADS=${JULIA_NUM_THREADS}"

# Auto-detect data file if requested, or if the provided path doesn't exist
if [[ "${AUTO_DATA}" -eq 1 || ! -f "${DATA}" ]]; then
  pattern="${DATA_PATTERN}"
  if [[ -z "${pattern}" ]]; then
    pattern="dataset/*bytes_1ms.csv"
  fi
  # Try preferred pattern first
  cand=$(ls -1t ${pattern} 2>/dev/null | head -n1 || true)
  # Try dataset/
  if [[ -z "${cand}" ]]; then
    cand=$(ls -1t dataset/*.csv 2>/dev/null | head -n1 || true)
  fi
  # Try traces/real_traces/
  if [[ -z "${cand}" ]]; then
    cand=$(ls -1t traces/real_traces/*.csv 2>/dev/null | head -n1 || true)
  fi
  # Try traces/synt_traces/fGn/
  if [[ -z "${cand}" ]]; then
    cand=$(ls -1t traces/synt_traces/fGn/*.csv 2>/dev/null | head -n1 || true)
  fi
  # Try traces/synt_traces/mwm/
  if [[ -z "${cand}" ]]; then
    cand=$(ls -1t traces/synt_traces/mwm/*.csv 2>/dev/null | head -n1 || true)
  fi
  if [[ -z "${cand}" ]]; then
    echo "[fatal] No CSV found for auto-detection (pattern='${pattern}')" >&2
    exit 2
  fi
  DATA="${cand}"
  echo "[auto-data] Selected DATA='${DATA}'"
fi

# Resolve output directory and default filename based on data file and L
TRACE_FILE="${DATA##*/}"              # e.g., 202301131400_bytes_1ms.csv
TRACE_BASE="${TRACE_FILE%.*}"        # e.g., 202301131400_bytes_1ms
OUT_DIR="results/${TRACE_BASE}/L_${MERA_L}"
if [[ "${OUTPUT_SET}" -eq 0 || -z "${OUTPUT}" ]]; then
  OUTPUT="${OUT_DIR}/train_${TRACE_BASE}.csv"
  if [[ "${DATE_STAMP}" -eq 1 ]]; then
    # Append today's date before extension, e.g., train_<trace>_YYYY-MM-DD.csv
    DATE_TAG="$(date +%F)"
    base="$(basename "${OUTPUT}")"
    dir="$(dirname "${OUTPUT}")"
    ext="${base##*.}"
    name="${base%.*}"
    if [[ "${base}" == "${ext}" ]]; then
      OUTPUT="${dir}/${name}_${DATE_TAG}"
    else
      OUTPUT="${dir}/${name}_${DATE_TAG}.${ext}"
    fi
  fi
fi
mkdir -p "$(dirname "${OUTPUT}")"

# Default models dir next to output if saving models and no custom dir provided
if [[ "${SAVE_MODELS}" -eq 1 && -z "${MODELS_DIR}" ]]; then
  MODELS_DIR="$(dirname "${OUTPUT}")/models"
fi
if [[ -n "${MODELS_DIR}" ]]; then
  mkdir -p "${MODELS_DIR}"
fi

echo "[run] data=${DATA} windows=${NUM_WINDOWS} start=${START_WINDOW} retains=${RETAINS} L=${MERA_L} chi=${MERA_CHI} chimid=${MERA_CHIMID} train=${DO_TRAIN} warm_start=${WARM_START} parallel_windows=${PARALLEL_WINDOWS}"
echo "[run] out_dir=${OUT_DIR} output=${OUTPUT}"

# Apply preset overrides for iterations
case "${PRESET}" in
  fast)
    STAGE1_ITERS=5; STAGE2_ITERS=5 ;;
  paper|default|"")
    # keep defaults (now 50/50)
    : ;;
  quality)
    STAGE1_ITERS=50; STAGE2_ITERS=50 ;;
  *)
    echo "[warn] Unknown preset '${PRESET}', ignoring" ;;
esac

# Thread pinning and BLAS env
if [[ "${PIN_THREADS}" == "1" ]]; then
  export JULIA_EXCLUSIVE=1
  # Align OpenBLAS threads with requested BLAS threads when available
  if [[ -n "${BLAS_THREADS}" ]]; then
    export OPENBLAS_NUM_THREADS="${BLAS_THREADS}"
  elif [[ "${PARALLEL_WINDOWS}" == "1" ]]; then
    export OPENBLAS_NUM_THREADS=1
  fi
  echo "[run] thread pinning enabled (JULIA_EXCLUSIVE=1, OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-unset})"

  # Probe if Julia accepts exclusive pinning with the current thread count.
  # If it fails (e.g., "Too many threads requested for JULIA_EXCLUSIVE option."),
  # decrement threads until it succeeds, or disable pinning as a fallback.
  if ! julia -e 'println(Threads.nthreads())' >/dev/null 2>&1; then
    ORIG_THREADS="${JULIA_NUM_THREADS}"
    T="${JULIA_NUM_THREADS}"
    while [[ "${T}" -gt 1 ]]; do
      T=$((T-1))
      export JULIA_NUM_THREADS="${T}"
      if julia -e 'println(Threads.nthreads())' >/dev/null 2>&1; then
        echo "[run] adjusted JULIA_NUM_THREADS=${T} for exclusive pinning (was ${ORIG_THREADS})"
        break
      fi
    done
    # If still failing at T==1, disable pinning and restore original threads
    if ! julia -e 'println(Threads.nthreads())' >/dev/null 2>&1; then
      echo "[warn] JULIA_EXCLUSIVE pinning failed with available CPUs; disabling pinning and restoring threads" >&2
      unset JULIA_EXCLUSIVE
      export JULIA_NUM_THREADS="${ORIG_THREADS}"
    fi
  fi
fi

if [[ "${PARALLEL_WINDOWS}" == "1" ]]; then
  # varMERA-style parallel runner (CPU-only). Map warm start to --init.
  INIT_ARG=$([[ "${WARM_START}" == "1" ]] && echo "--init=haar" || echo "--init=random")
  CMD=(
    julia scripts/run_varmera_style_optimization.jl
    --data "${DATA}"
    --window-size="${WINDOW_SIZE}" --step="${STEP}"
    --num-windows="${NUM_WINDOWS}" --start-window="${START_WINDOW}"
    --retains "${RETAINS}"
    --mera-L="${MERA_L}" --mera-chi="${MERA_CHI}" --mera-chimid="${MERA_CHIMID}"
    --stage1-iters="${STAGE1_ITERS}" --stage1-lr="${STAGE1_LR}"
    --stage2-iters="${STAGE2_ITERS}" --stage2-lr="${STAGE2_LR}"
    --baseline-waves "${BASELINE_WAVES}"
  --output "${OUTPUT}"
  --param "${PARAM}"
    $([[ "${EARLY_STOP}" == "1" ]] && echo "--early-stop")
    --tol="${TOL}" --patience="${PATIENCE}" --min-iter="${MIN_ITER}"
    --sparsity-weight="${SPARSITY_WEIGHT}" --mse-weight="${MSE_WEIGHT}"
    --parallel-windows --disable-intra-window
    ${INIT_ARG}
  )
  # Optional model saving flags for the parallel runner
  if [[ "${SAVE_MODELS}" -eq 1 ]]; then
    CMD+=( --save-models )
  fi
  if [[ -n "${MODELS_DIR}" ]]; then
    CMD+=( --models-dir="${MODELS_DIR}" )
  fi
  if [[ -n "${SEED}" ]]; then
    CMD+=( --seed="${SEED}" )
  fi
  # BLAS threads control (default to 1 if not provided)
  if [[ -n "${BLAS_THREADS}" ]]; then
    CMD+=( --blas-threads="${BLAS_THREADS}" )
  else
    CMD+=( --blas-threads=1 )
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[dry-run] '; printf '%q ' "${CMD[@]}"; echo
    exit 0
  fi
  "${CMD[@]}"
else
  # Clean sequential runner
  CMD=(
    julia scripts/mera_mawi_runner.jl
    --data "${DATA}"
    --window-size="${WINDOW_SIZE}" --step="${STEP}"
    --num-windows="${NUM_WINDOWS}" --start-window="${START_WINDOW}"
    --retains "${RETAINS}"
    --mera-L="${MERA_L}" --mera-chi="${MERA_CHI}" --mera-chimid="${MERA_CHIMID}"
    --stage1-iters="${STAGE1_ITERS}" --stage1-lr="${STAGE1_LR}"
    --stage2-iters="${STAGE2_ITERS}" --stage2-lr="${STAGE2_LR}"
    --baseline-waves "${BASELINE_WAVES}"
    --output "${OUTPUT}"
  )
  if [[ -n "${SEED}" ]]; then
    CMD+=( --seed="${SEED}" )
  fi
  if [[ "${WARM_START}" == "1" ]]; then
    CMD+=( --warm-start-haar )
  fi
  if [[ "${DO_TRAIN}" == "1" ]]; then
    CMD+=( --train )
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[dry-run] '; printf '%q ' "${CMD[@]}"; echo
    exit 0
  fi
  "${CMD[@]}"
fi

echo "[done] Wrote results to ${OUTPUT}"

# Post-run analysis: periodogram (Welch and Daniell) and multiscale aggregation
if [[ "${DO_PERIODOGRAM}" == "1" ]]; then
  echo "[post] Generating smoothed periodograms (welch + daniell)"
  # Base prefix in results/<trace>/L_<L>/
  TRACE_FILE="${DATA##*/}"
  TRACE_BASE="${TRACE_FILE%.*}"
  BASE_DIR="results/${TRACE_BASE}/L_${MERA_L}"
  mkdir -p "${BASE_DIR}"
  if ! julia --project=. scripts/analysis/plot_smoothed_periodogram.jl \
    --data "${DATA}" --fs 1000 --method welch --segment-length 4096 --overlap 0.5 --window hann \
    --fit-frac 0.1 --title "${TRACE_BASE}" --out-prefix "${BASE_DIR}/periodogram_L_${MERA_L}"; then
    echo "[post][warn] welch periodogram failed (non-fatal)" >&2
    if [[ "${POST_STRICT}" == "1" ]]; then echo "[post][fatal] aborting due to --post-strict" >&2; exit 1; fi
  fi
  if ! julia --project=. scripts/analysis/plot_smoothed_periodogram.jl \
    --data "${DATA}" --fs 1000 --method daniell --segment-length 4096 --overlap 0.5 --window hann \
    --fit-frac 0.1 --title "${TRACE_BASE}" --out-prefix "${BASE_DIR}/periodogram_daniell_L_${MERA_L}"; then
    echo "[post][warn] daniell periodogram failed (non-fatal)" >&2
    if [[ "${POST_STRICT}" == "1" ]]; then echo "[post][fatal] aborting due to --post-strict" >&2; exit 1; fi
  fi
fi

if [[ "${DO_AGG}" == "1" ]]; then
  echo "[post] Generating multiscale aggregation view"
  TRACE_FILE="${DATA##*/}"
  TRACE_BASE="${TRACE_FILE%.*}"
  BASE_DIR="results/${TRACE_BASE}/L_${MERA_L}"
  mkdir -p "${BASE_DIR}"
  if ! julia --project=. scripts/analysis/plot_multiscale_aggregation.jl \
    --data "${DATA}" --fs 1000 --scales-ms 1,2,4,8,16 --agg mean --max-points 4000 --title "${TRACE_BASE}" \
    --out "${BASE_DIR}/multiscale_aggregation_L_${MERA_L}.png"; then
    echo "[post][warn] multiscale aggregation plot failed (non-fatal)" >&2
    if [[ "${POST_STRICT}" == "1" ]]; then echo "[post][fatal] aborting due to --post-strict" >&2; exit 1; fi
  fi
fi

# Post-run analysis: Wavelet baselines (PSNR) for requested set
if [[ "${DO_BASELINE_WAVES}" == "1" ]]; then
  echo "[post] Computing baseline DWT PSNRs: ${BASELINE_WAVES}"
  TRACE_FILE="${DATA##*/}"
  TRACE_BASE="${TRACE_FILE%.*}"
  OUT_DIR="results/${TRACE_BASE}/L_${MERA_L}"
  mkdir -p "${OUT_DIR}"
  BASE_CSV="${OUT_DIR}/wavelet_baseline_psnr.csv"
  MERA_CSV="${OUT_DIR}/train_${TRACE_BASE}.csv"
  if ! julia --project=. scripts/analysis/baseline_wavelet_psnr.jl \
    --data "${DATA}" --window-size="${WINDOW_SIZE}" --step="${STEP}" \
    --start-window="${START_WINDOW}" --num-windows="${NUM_WINDOWS}" \
    --retains "${RETAINS}" --waves "${BASELINE_WAVES}" --mera-L="${MERA_L}" \
    --output "${BASE_CSV}"; then
    echo "[post][warn] baseline wavelet PSNR computation failed (non-fatal)" >&2
    if [[ "${POST_STRICT}" == "1" ]]; then echo "[post][fatal] aborting due to --post-strict" >&2; exit 1; fi
  fi
  # Gerar plots PSNR e ΔPSNR
  echo "[post] Gerando plots PSNR/ΔPSNR"
  julia --project=. scripts/analysis/plot_psnr_comparison.jl \
    --mera "${MERA_CSV}" \
    --baseline "${BASE_CSV}" \
    --waves "${BASELINE_WAVES}" \
    --out1 "${OUT_DIR}/psnr_vs_retain_L_${MERA_L}.png" \
    --out2 "${OUT_DIR}/psnr_gain_vs_retain_L_${MERA_L}.png"
fi

# Post-run analysis: ΔH vs retain (Hurst preservation)
if [[ "${DO_HURST}" == "1" ]]; then
  echo "[post] Generating ΔH vs retain (Hurst preservation)"
  TRACE_FILE="${DATA##*/}"
  TRACE_BASE="${TRACE_FILE%.*}"
  OUT_DIR="results/${TRACE_BASE}/L_${MERA_L}"
  mkdir -p "${OUT_DIR}"
  MERA_CSV="${OUT_DIR}/train_${TRACE_BASE}.csv"
  HCSV="${OUT_DIR}/hurst_compare_retains.csv"
  # Backward compatibility: mirror main CSV as hurst_compare_retains.csv
  cp "${MERA_CSV}" "${HCSV}"
  # Summarize and plot (uses MERA CSV directly)
  if ! julia --project=. scripts/analysis/plot_hurst_preservation_vs_retain.jl \
    --csv "${MERA_CSV}" \
    --title "ΔH vs retain — ${TRACE_BASE} L=${MERA_L} (95% CI)"; then
    echo "[post][warn] ΔH plot failed (non-fatal)" >&2
    if [[ "${POST_STRICT}" == "1" ]]; then echo "[post][fatal] aborting due to --post-strict" >&2; exit 1; fi
  fi
fi
