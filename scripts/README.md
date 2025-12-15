# Scripts overview

This folder has many helpers. For the 1D MERA run via the wrapper, only the scripts in the root are required:

Core (required for run_mera_mawi.sh)
- run_mera_mawi.sh — generic wrapper (accepts flags; parallel windows ON by default)
- mera_mawi_runner.jl — clean sequential runner: loads data, trains/analyzes, writes CSV
- run_varmera_style_optimization.jl — parallel windows runner (used when --parallel-windows)

Everything else is optional and grouped below for reference.

Analysis / plotting (optional) — scripts/analysis/
- baseline_wavelet_psnr.jl — compute Haar/DB4 baselines
- merge_psnr_wide.jl — join MERA + baselines into a wide CSV
- plot_psnr_comparison.jl — PSNR vs retain and MERA gains
- plot_psnr_correlations.jl — PSNR vs Hurst/alpha_width
- plot_robustness_vs_multifractal.jl — ΔPSNR vs multifractal metrics
- summarize_parallel_metrics.jl — summarize MERA CSV means and timing
- analyze_compression_impact.jl, analyze_mawi_metrics.jl, compare_hurst_retains.jl, plot_wavelet_hurst_spectrum.jl — exploratory utilities

Benchmarks / tests (optional) — scripts/benchmarks/
- benchmark_experiment.sh, benchmark_cuda_performance.sh, test_cuda_parallel.sh, test_mera_cuda.sh
- test_learn_driver_gpu.jl

Legacy / orchestration (optional) — scripts/legacy/
- run_mawi_experiments.jl, run_parallel_experiments.sh, run_optimized_experiments.sh
- monitor_experiments.sh, distributed_experiments.sh, launch_distributed.sh
- backups: run_mawi_experiments.corrupted.bak, _run_mawi_experiments.jl

Notes
- Prefer run_mera_mawi.sh (wrapper) — it handles threads, BLAS, seed, and mode selection for you.
- For paper pipeline, run the wrapper, then analysis in scripts/analysis as needed.