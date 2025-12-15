#!/bin/bash

# Benchmark rápido: medir tempo para 1 janela
echo "🚀 BENCHMARK RÁPIDO: 1 janela com retain_ratio = 0.01 (FORÇANDO CUDA)"
echo "=========================================="
echo "Iniciando em: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Progresso: [🔄] Executando..."

start_time=$(date +%s)

# Forçar uso do CUDA
export JULIA_CUDA_FORCE=true

julia scripts/run_mawi_experiments.jl \
    --step=4096 \
    --num-windows=1 \
    --retain=0.01 \
    --output="results/benchmark_1_window_cuda.csv"

end_time=$(date +%s)
duration=$((end_time - start_time))

echo "Progresso: [✅] Concluído!"
echo ""
echo "⏱️  Tempo para 1 janela: ${duration} segundos"
echo "📊 Estimativa para 100 janelas: ~$((duration * 100 / 60)) minutos"
echo "📊 Estimativa para 100 janelas: ~$((duration * 100 / 3600)) horas"
echo ""
echo "✅ Benchmark concluído! Arquivo: results/benchmark_1_window_cuda.csv"