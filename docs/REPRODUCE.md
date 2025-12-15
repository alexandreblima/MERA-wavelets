# MERA — Reprodutibilidade (paper-ready)

Este guia descreve como reproduzir os principais experimentos, métricas e figuras do projeto.

## Pré-requisitos
- Julia 1.10/1.11 (conforme `Project.toml`)
- Este repositório clonado e com o ambiente ativado (o wrapper já cuida disso)

## TL;DR (o caminho feliz)
1) Dê permissão ao wrapper e rode o smoke (rápido, CPU-only com paralelismo entre janelas):
   ```bash
   chmod +x scripts/run_mera_mawi.sh
   scripts/run_mera_mawi.sh \
     --data=data/small_test.csv \
     --window-size=256 --step=256 --num-windows=2 \
     --retains=0.01,0.02,0.05,0.1 \
     --mera-L=3 --mera-chi=2 --mera-chimid=2 \
     --preset=fast --warm-start-haar --train --pin-threads \
     --output=results/_task_smoke.csv
   ```
2) Execute o modo “paper-ready” (paralelo entre janelas, seed fixo):
   ```bash
   scripts/run_mera_mawi.sh \
     --data=data/MAWI_bytes_1ms.csv \
     --window-size=1024 --step=1024 --num-windows=10 \
     --retains=0.01,0.02,0.05,0.1,0.2,0.4,0.8 \
     --mera-L=5 --mera-chi=2 --mera-chimid=2 \
     --preset=paper --seed=12345 --warm-start-haar --pin-threads \
     --output=results/_task_parallel_paper.csv
   ```
3) Gere/analise figuras e tabelas em `scripts/analysis/` (ver seção “Figuras e Tabelas”).

## Parâmetros “paper-ready” (1D)
- MERA: `L=5`, `chi=2`, `chimid=2`
- Retains: `0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.80`
- Presets de iterações: `--preset=paper` (equivale a Stage1=25, Stage2=25; `fast` usa 5/5 e `quality` usa 50/50)
- Paralelismo: inter-janelas habilitado por padrão; BLAS=1 nesse modo (pode ajustar com `--blas-threads`)
- Reprodutibilidade: `--seed=12345` (CSV inclui `git_commit`, `git_dirty`, `manifest_sha256` e `seed` por linha + linha resumo `window_id=0`)

## Entradas e saídas
- Entrada principal: `data/MAWI_bytes_1ms.csv`
- Saídas (CSV) organizadas por trace e nível, p.ex.: `results/<trace_base>/L_<L>/train_<trace_base>.csv`, com métricas por janela/retain e linha resumo (`window_id=0`).
- PSNR por retain; H; e, quando habilitado, ΔH vs retain (seção dedicada abaixo).
- Tempo total do job (linha `window_id=0`).

## Como rodar via VS Code
- Command Palette → “Run Task…”:
  - “Run: MERA MAWI (smoke)”
  - “Run: MERA MAWI (parallel, paper)”

## Figuras e Tabelas (mapeamento)
– Comparações PSNR vs retain (baselines × MERA), salvas ao lado do CSV principal e com sufixo de nível quando aplicável:
  - `psnr_vs_retain_L_<L>.{png,svg,pdf}`
  - `psnr_gain_vs_retain_L_<L>.{png,svg,pdf}`
– ΔPSNR vs CR (CR = 1/retain), L=5:
  - `psnr_L_L5_psnr_vs_cr.{png,svg,pdf}`
  - `psnr_L_L5_gain_vs_cr.{png,svg,pdf}`
– Periodogramas e agregação multiescala (organizados por L):
  - `results/<trace>/L_<L>/periodogram_L_<L>.*`, `results/<trace>/L_<L>/periodogram_daniell_L_<L>.*`, `results/<trace>/L_<L>/multiscale_aggregation_L_<L>.*`
– ΔH vs retain (95% CI), organizados na estrutura L:
  - `results/<trace>/L_<L>/hurst_preservation_L_<L>.(png|pdf|svg)` e `results/<trace>/L_<L>/hurst_preservation_L_<L>_summary.csv`
– CSVs auxiliares “wide” e correlações (opcionais, para estudos):
  - `scripts/analysis/merge_psnr_wide.jl`, `scripts/analysis/plot_psnr_correlations.jl`

Para gerar/atualizar:
- Baselines Haar/DB4: `julia --project=. scripts/analysis/baseline_wavelet_psnr.jl --data <csv> --window-size <n> --step <n> --num-windows <k> --mera-L <L>`
- Plots PSNR/gains (salva ao lado do CSV do MERA): `julia --project=. scripts/analysis/plot_psnr_comparison.jl --mera <mera_csv> --baseline <baseline_csv>`
- ΔPSNR vs CR (L=5): `julia --project=. scripts/analysis/plot_psnr_gain_lfiltered.jl --mera <mera_csv> --baseline <baseline_csv> --L 5`
- Periodogramas: `julia --project=. scripts/analysis/plot_smoothed_periodogram.jl --data <csv> --method welch|daniell`
- Agregação multiescala: `julia --project=. scripts/analysis/plot_multiscale_aggregation.jl --data <csv>`
- ΔH vs retain (pipeline manual):
  1) `julia --project=. scripts/analysis/compare_hurst_retains.jl --data <csv> --mera-L <L> --output <out_csv>`
  2) `julia --project=. scripts/analysis/plot_hurst_preservation_vs_retain.jl --csv <out_csv>`
- Resumo de tempos por retain: `julia --project=. scripts/analysis/summarize_parallel_metrics.jl --input <mera_csv> --output <out_csv>`

## Paralelismo e performance (CPU-only)
- Paralelismo entre janelas (Threads Julia) é o caminho principal. O wrapper autodetecta `JULIA_NUM_THREADS` (ou defina manualmente) e, com `--pin-threads`, usa `JULIA_EXCLUSIVE=1` e ajusta BLAS.
- BLAS em 1 quando paralelizando várias janelas costuma ser melhor. Ajuste com `--blas-threads=N` se necessário.
- CUDA/GPU foram removidos do núcleo; não há dependência de GPU.

## Troubleshooting
- “Permissão negada” no wrapper: rode `chmod +x scripts/run_mera_mawi.sh`.
- Dependências Julia: ao rodar qualquer script `.jl`, o projeto `Project.toml` é ativado; o primeiro run baixa/compila dependências.
- Reprodutibilidade: confira na linha resumo do CSV os campos `git_commit`, `git_dirty`, `manifest_sha256` e `seed`.

## Testes
- Modo rápido (smoke):
  ```bash
  FAST_TESTS=1 WAVE6G_TEST_ITERS=10 julia --project=. test/runtests.jl
  ```
- Modo completo:
  ```bash
  julia --project=. test/runtests.jl
  ```
Observações: o modo rápido reduz iterações do aprendizado e acelera validações locais/CI, mantendo as verificações de queda de loss e reconstrução determinística.

---

Dúvidas? Abra uma issue no GitHub ou mencione o arquivo/CSV/figura específica.
