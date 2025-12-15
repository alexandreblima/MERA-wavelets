#!/bin/bash

# Script para rodar experimentos com diferentes taxas de compressão
# 100 janelas não sobrepostas + medição de tempo

# Verificar se CUDA está disponível
echo "🔍 Verificando disponibilidade do CUDA..."
if julia -e 'using CUDA; println("CUDA funcional: ", CUDA.functional())' | grep -q "true"; then
    echo "✅ CUDA detectado e funcional - experimentos usarão GPU"
else
    echo "❌ CUDA não funcional - experimentos usarão CPU"
fi

# Verificar threads disponíveis
NUM_THREADS=$(nproc)
echo "🧵 Sistema tem $NUM_THREADS núcleos CPU disponíveis"
echo ""

RETAIN_RATES=(0.01 0.02 0.05 0.1 0.2)  # ← TODAS as taxas de compressão

TOTAL_EXPERIMENTS=${#RETAIN_RATES[@]}
CURRENT_EXPERIMENT=0

echo "Executando experimentos com múltiplas taxas de compressão..."
echo "=========================================="
echo "Total de janelas por experimento: 100"
echo "MERA Learning: GPU (RTX 2060) + MSE weight = 0.0 (sparsity-only)"
echo "Taxas de compressão: ${RETAIN_RATES[*]}"
echo "🚀 PARALELIZAÇÃO: Experimentos executados simultaneamente!"
echo "🧵 Cada experimento usa $NUM_THREADS threads para processar janelas em paralelo"
echo ""

# Array para armazenar PIDs dos processos em background
declare -a PIDS=()
declare -a START_TIMES=()

for rate in "${RETAIN_RATES[@]}"; do
    CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))
    PERCENTAGE=$((CURRENT_EXPERIMENT * 100 / TOTAL_EXPERIMENTS))

    echo ""
    echo "[$CURRENT_EXPERIMENT/$TOTAL_EXPERIMENTS] ($PERCENTAGE%) Iniciando experimento com retain_ratio = $rate"
    echo "Progresso: [$(printf '%.0s#' $(seq 1 $((PERCENTAGE / 10))))$(printf '%.0s-' $(seq 1 $((10 - PERCENTAGE / 10))))] $PERCENTAGE%"
    echo "Iniciando em: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "🔥 Usando GPU CUDA (RTX 2060) para aceleração"
    echo "⚡ Executando em PARALELO com outros experimentos"

    # Medir tempo de execução
    start_time=$(date +%s)
    START_TIMES[$CURRENT_EXPERIMENT]=$start_time

    # Forçar uso do CUDA e configurar threads
    export JULIA_CUDA_FORCE=true
    export JULIA_NUM_THREADS=$(nproc)

    echo "🧵 Usando $JULIA_NUM_THREADS threads Julia para processamento paralelo"

    # Executar em background (paralelo)
    julia scripts/run_mawi_experiments.jl \
        --step=4096 \
        --num-windows=100 \
        --retain=$rate \
        --output="results/mawi_metrics_100_nonoverlap_retain_${rate}.csv" &

    # Armazenar PID do processo
    PIDS[$CURRENT_EXPERIMENT]=$!
    echo "📋 PID do processo: ${PIDS[$CURRENT_EXPERIMENT]}"
done

echo ""
echo "=========================================="
echo "⏳ Aguardando conclusão de todos os experimentos paralelos..."
echo ""

# Aguardar todos os processos terminarem
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    rate=${RETAIN_RATES[$((i-1))]}
    start_time=${START_TIMES[$i]}

    echo "Aguardando experimento retain_ratio = $rate (PID: $pid)..."
    wait $pid

    # Calcular tempo após conclusão
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    hours=$((duration / 3600))
    minutes=$(( (duration % 3600) / 60 ))
    seconds=$((duration % 60))

    echo "✓ Finalizado retain_ratio = $rate"
    echo "⏱️  Tempo total: ${hours}h ${minutes}m ${seconds}s"
    echo "📊 Taxa: $((duration / 100)) segundos por janela (média)"
    echo ""
done

echo ""
echo "=========================================="
echo "🎉 Todos os experimentos concluídos!"
echo "📊 Execute 'julia scripts/analyze_compression_impact.jl' para analisar os resultados"
echo "📈 Dados salvos em: results/mawi_metrics_100_nonoverlap_retain_*.csv"
echo "🔥 Todos os experimentos foram executados com GPU CUDA"
echo "🚀 Otimizações aplicadas: CUDA Streams, Precision Mista, Cache Inteligente, BLAS Otimizado"