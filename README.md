### Wave6G — MERA Pipeline for Traffic Traces

Wave6G adapts the variational **MERA** (Multi-scale Entanglement Renormalization Ansatz) idea to study long-range dependent network traces. The toolkit trains MERA networks on CSV time-series, benchmarks against classical wavelet families, and produces diagnostic plots (PSNR, ΔH, periodograms, etc.).

---

## Quickstart

Train and analyse a MAWI-like trace in one shot (parallel CPU runner):

```bash
chmod +x scripts/run_mera_mawi.sh

scripts/run_mera_mawi.sh \
  --data=dataset/202301131400_bytes_1ms.csv \
  --window-size=4096 --step=4096 \
  --num-windows=0 --start-window=1 \
  --mera-L=5 --mera-chi=2 --mera-chimid=2 \
  --preset=paper --seed=12345 \
  --warm-start-haar --hurst-preservation
```

Outputs are placed in `results/<trace>/L_<L>/`, e.g.

```
results/202301131400_bytes_1ms/L_5/
    train_202301131400_bytes_1ms.csv
    wavelet_baseline_psnr.csv
    psnr_vs_retain_L_5.(png|pdf|svg)
    psnr_gain_vs_retain_L_5.(png|pdf|svg)
    hurst_preservation_L_5.(png|pdf|svg)
    hurst_preservation_L_5_summary.csv
    periodogram_L_5.(png|pdf|svg)
    periodogram_daniell_L_5.(png|pdf|svg)
    multiscale_aggregation_L_5.(png|pdf|svg)
```

- `train_*.csv` contains per-window metrics (PSNR, ΔH, etc.) plus metadata (`window_id = 0` row).
- Baseline CSV holds Haar/DB4/C3/S8/Bior4.4 PSNRs; the runner always ensures this canonical set is present.
- All PSNR and ΔH plots now include **block-bootstrap intervals**, controllable via flags (see below).

---

## What the Runner Does

`scripts/run_mera_mawi.sh` orchestrates the entire pipeline:

1. **Pre-flight**  
   Normalises options, activates the repo project, sets `JULIA_NUM_THREADS`, resolves default retain ratios and output paths, and (optionally) autodetects the newest CSV in `dataset/`.

2. **Diagnostics on raw trace** (non-fatal if they fail)  
   - `scripts/analysis/diagnose_multifractality.jl` — multifractal spectrum & diagnosis.  
   - `scripts/analysis/compute_hurst_ci.jl` — Hurst estimate with CI on the original trace.

3. **MERA training**  
   Delegates to `scripts/run_varmera_style_optimization.jl`, using inter-window parallelism by default. Warm starts (Haar vs random), presets (`fast|paper|quality`), early stopping, sparsity weights, and alternate parametrisations (`--param=angle`) are exposed through wrapper flags.

4. **Post-processing**  
   - Computes Welch & Daniell periodograms.  
   - Generates a multiscale aggregation view.  
   - Runs classical wavelet baselines (`baseline_wavelet_psnr.jl`).  
   - Produces PSNR/ΔPSNR plots via `plot_psnr_comparison.jl`.  
   - If `--hurst-preservation` (default), mirrors `train_*.csv` to `hurst_compare_retains.csv` and emits ΔH vs retain plots with `plot_hurst_preservation_vs_retain.jl`.

Each plotting script emits PNG/PDF/SVG. `--post-strict` makes post-processing failures abort the run; otherwise the training CSV is still written.

---

## Block-Bootstrap Confidence Intervals

Both plotting scripts now accept a consistent trio of flags:

```
--bootstrap-block <int>      # block length (in windows) used for resampling (default: 8)
--bootstrap-samples <int>    # number of bootstrap replicates (default: 2000; 0 => fallback to z·σ/√n)
--bootstrap-seed <int>       # seed >= 0 to make bootstrap reproducible
```

### PSNR Plots

```bash
julia --project=. scripts/analysis/plot_psnr_comparison.jl \
  --mera results/.../train_*.csv \
  --baseline results/.../wavelet_baseline_psnr.csv \
  --waves haar,db4,c3,s8,bior4.4 \
  --bootstrap-block 8 \
  --bootstrap-samples 4000
```

- The script aligns MERA and wavelet PSNRs by `window_id`, resamples them in blocks, and provides asymmetric `ci_low/ci_high`.  
- ΔPSNR curves are computed on the paired differences before bootstrapping, preserving the dependency between methods.

### ΔH (Hurst Preservation)

```bash
julia --project=. scripts/analysis/plot_hurst_preservation_vs_retain.jl \
  --csv results/.../train_*.csv \
  --bootstrap-block 8 \
  --bootstrap-samples 4000
```

- Uses the columns `deltaH_*` already stored in the MERA CSV (and copied to `hurst_compare_retains.csv` by the runner).  
- Summary CSV now exposes `_ci_low/_ci_high` columns per series; update downstream tooling accordingly.

**Choosing the block size**  
Start with 8–12 windows; increase it if PSNR/H curves still show strong serial correlation. When unsure, inspect the autocorrelation of the per-window metrics and select a block slightly longer than the correlation range.

---

## Batch Run for Multiple Levels

To sweep several `L` levels with 1024-sample windows, wrap the runner in a simple shell loop. The snippet below executes the full pipeline (diagnostics, baselines, ΔH plots) for `L=3` through `L=8` on the same trace:

```bash
chmod +x scripts/run_mera_mawi.sh

TRACE=dataset/202301131400_bytes_1ms.csv
for L in 3 4 5 6 7 8; do
  scripts/run_mera_mawi.sh \
    --data="${TRACE}" \
    --window-size=1024 --step=1024 \
    --num-windows=0 --start-window=1 \
    --mera-L=${L} --mera-chi=2 --mera-chimid=2 \
    --preset=paper --seed=12345 \
    --warm-start-haar --hurst-preservation
done
```

- Tweak `--mera-chi/--mera-chimid` if you want to compare different bond dimensions.
- For repeatable runs, keep `--seed` fixed and optionally set `--bootstrap-seed` during plotting.
- Outputs land in `results/<trace_base>/L_<L>/`, each containing the corresponding CSVs and figures.

To parallelise, launch one `L` per terminal/job; the runner isolates outputs by directory, so files do not clash.

---

## Additional Utilities

- `scripts/analysis/estimate_hurst_raw.jl` – DFA + bootstrap on the raw trace (optionally Local Whittle).  
- `scripts/analysis/plot_smoothed_periodogram.jl` – Welch, Daniell or log-binned spectra (`--method`).  
- `scripts/analysis/plot_multiscale_aggregation.jl` – time-domain aggregation plot across scales.  
- `scripts/analysis/baseline_wavelet_psnr.jl` – recompute PSNR baselines without rerunning MERA (supports custom retain sets and wavelet lists).  
- `scripts/analysis/plot_psnr_gain_lfiltered.jl` – PSNR vs compression ratio plots for a fixed level `L`.

All scripts honour `JULIA_PROJECT` and will write outputs next to the referenced CSV unless `--out`/`--out-prefix` is provided.

---

## Tips & Troubleshooting

- **Overlapping windows**: if you decrease `--step`, increase `--bootstrap-block` to capture the longer correlation.  
- **Periodogram fits**: `--fit-frac` controls the fraction of low frequencies used to fit the slope.  
- **Baseline auto-completion**: even if you pass a subset via `--baseline-waves`, the runner adds the canonical `{haar, db4, c3, s8, bior4.4}`.  
- **Dry runs**: `scripts/run_mera_mawi.sh --dry-run ...` prints the resolved `julia` command without executing.  
- **Reproducibility**: fix `--seed` and optionally `--bootstrap-seed` to regenerate identical training and CI bands.

For a deeper look at experiment batches, see `docs/REPRODUCE.md` and `scripts/README.md`.
