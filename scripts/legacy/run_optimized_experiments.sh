#!/bin/bash

# Script melhorado para rodar experimentos com paralelização otimizada
# Controle de recursos e gerenciamento inteligente de processos

# Configurações de paralelização
MAX_CONCURRENT_EXPERIMENTS=${MAX_CONCURRENT_EXPERIMENTS:-2}  # Máximo de experimentos simultâneos
MAX_MEMORY_PERCENT=${MAX_MEMORY_PERCENT:-80}                # Máximo de memória RAM (%)
GPU_MEMORY_CHECK=${GPU_MEMORY_CHECK:-true}                   # Verificar memória GPU

# Verificar disponibilidade do CUDA
echo "🔍 Verificando disponibilidade do CUDA..."
if julia -e 'using CUDA; println("CUDA funcional: ", CUDA.functional())' | grep -q "true"; then
    echo "✅ CUDA detectado e funcional - experimentos usarão GPU"
    HAS_CUDA=true
else
    echo "❌ CUDA não funcional - experimentos usarão CPU"
    HAS_CUDA=false
fi

# Verificar recursos do sistema
NUM_CORES=$(nproc)
TOTAL_MEMORY_GB=$(free -g | awk 'NR==2{printf "%.0f", $2}')
echo "🧵 Sistema tem $NUM_CORES núcleos CPU e ${TOTAL_MEMORY_GB}GB RAM"

# Detectar e configurar GPU
if $HAS_CUDA; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | sed 's/ MiB//')
    GPU_MEMORY_GB=$((GPU_MEMORY / 1024))

    echo "🎮 GPU detectada: $GPU_NAME (${GPU_MEMORY_GB}GB)"

    # Configurar streams baseado na GPU
    if [[ $GPU_NAME == *"RTX 2060"* ]]; then
        # RTX 2060: 6GB VRAM, boa para 4-6 streams
        JULIA_CUDA_STREAMS=6
        echo "🔄 RTX 2060 detectada - configurando 6 streams CUDA"
    elif [[ $GPU_NAME == *"RTX 30"* ]] || [[ $GPU_NAME == *"RTX 40"* ]]; then
        # RTX 30/40 series: mais VRAM, mais streams
        JULIA_CUDA_STREAMS=8
        echo "🔄 RTX 30/40 series detectada - configurando 8 streams CUDA"
    else
        # GPUs genéricas
        JULIA_CUDA_STREAMS=4
        echo "🔄 GPU genérica detectada - configurando 4 streams CUDA"
    fi

    export JULIA_CUDA_STREAMS
else
    echo "💻 Usando apenas CPU"
fi
echo ""

# Função para verificar uso de memória
check_memory_usage() {
    local mem_percent=$(free | awk 'NR==2{printf "%.0f", $3*100/$2}')
    echo $mem_percent
}

# Função para verificar memória GPU
check_gpu_memory() {
    if $HAS_CUDA && $GPU_MEMORY_CHECK; then
        local gpu_mem_used=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk -F',' '{print $1}')
        local gpu_mem_total=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk -F',' '{print $2}')
        local gpu_mem_percent=$((gpu_mem_used * 100 / gpu_mem_total))
        echo $gpu_mem_percent
    else
        echo 0
    fi
}

# Função para aguardar recursos disponíveis
wait_for_resources() {
    local experiment_name="$1"
    echo "⏳ Aguardando recursos para: $experiment_name"

    while true; do
        local mem_usage=$(check_memory_usage)
        local gpu_mem_usage=$(check_gpu_memory)

        if [ $mem_usage -lt $MAX_MEMORY_PERCENT ] && [ $gpu_mem_usage -lt 90 ]; then
            echo "✅ Recursos disponíveis - Iniciando $experiment_name"
            break
        fi

        echo "🔄 Recursos insuficientes (RAM: ${mem_usage}%, GPU: ${gpu_mem_usage}%) - aguardando..."
        sleep 30
    done
}

RETAIN_RATES=(0.01 0.02 0.05 0.1 0.2)
TOTAL_EXPERIMENTS=${#RETAIN_RATES[@]}
CURRENT_EXPERIMENT=0

echo "🚀 EXPERIMENTOS COM PARALELIZAÇÃO OTIMIZADA"
echo "=========================================="
echo "Total de experimentos: $TOTAL_EXPERIMENTS"
echo "Máximo simultâneo: $MAX_CONCURRENT_EXPERIMENTS"
echo "Controle de memória: ${MAX_MEMORY_PERCENT}% RAM máxima"
echo "Taxas de compressão: ${RETAIN_RATES[*]}"
echo ""

# Array para armazenar processos ativos
declare -a ACTIVE_PIDS=()
declare -a ACTIVE_RATES=()
declare -a START_TIMES=()

# Função para gerenciar fila de processos
manage_process_queue() {
    # Remover processos concluídos
    local i=0
    while [ $i -lt ${#ACTIVE_PIDS[@]} ]; do
        local pid=${ACTIVE_PIDS[$i]}
        if ! kill -0 $pid 2>/dev/null; then
            # Processo terminou
            local rate=${ACTIVE_RATES[$i]}
            local start_time=${START_TIMES[$i]}
            local end_time=$(date +%s)
            local duration=$((end_time - start_time))

            echo "✅ Experimento concluído: retain_ratio = $rate (${duration}s)"

            # Remover da lista
            unset ACTIVE_PIDS[$i]
            unset ACTIVE_RATES[$i]
            unset START_TIMES[$i]
            ACTIVE_PIDS=("${ACTIVE_PIDS[@]}")
            ACTIVE_RATES=("${ACTIVE_RATES[@]}")
            START_TIMES=("${START_TIMES[@]}")
        else
            i=$((i + 1))
        fi
    done
}

# Executar experimentos com controle de recursos
for rate in "${RETAIN_RATES[@]}"; do
    CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))
    PERCENTAGE=$((CURRENT_EXPERIMENT * 100 / TOTAL_EXPERIMENTS))

    echo ""
    echo "[$CURRENT_EXPERIMENT/$TOTAL_EXPERIMENTS] ($PERCENTAGE%) Preparando experimento: retain_ratio = $rate"

    # Aguardar slot disponível
    while [ ${#ACTIVE_PIDS[@]} -ge $MAX_CONCURRENT_EXPERIMENTS ]; do
        manage_process_queue
        if [ ${#ACTIVE_PIDS[@]} -ge $MAX_CONCURRENT_EXPERIMENTS ]; then
            echo "🔄 Fila cheia (${#ACTIVE_PIDS[@]}/$MAX_CONCURRENT_EXPERIMENTS) - aguardando liberação..."
            sleep 10
        fi
    done

    # Aguardar recursos disponíveis
    wait_for_resources "retain_ratio = $rate"

    # Iniciar experimento
    echo "🔥 Iniciando retain_ratio = $rate"
    start_time=$(date +%s)

    # Configurar ambiente
    export JULIA_CUDA_FORCE=true
    export JULIA_NUM_THREADS=$NUM_CORES

    # Executar em background
    julia scripts/run_mawi_experiments.jl \
        --step=4096 \
        --num-windows=100 \
        --retain=$rate \
        --output="results/optimized_retain_${rate}.csv" &

    pid=$!
    ACTIVE_PIDS+=($pid)
    ACTIVE_RATES+=($rate)
    START_TIMES+=($start_time)

    echo "📋 PID: $pid | Processos ativos: ${#ACTIVE_PIDS[@]}/$MAX_CONCURRENT_EXPERIMENTS"
done

# Aguardar todos os experimentos terminarem
echo ""
echo "⏳ Aguardando conclusão de todos os experimentos..."
while [ ${#ACTIVE_PIDS[@]} -gt 0 ]; do
    manage_process_queue
    if [ ${#ACTIVE_PIDS[@]} -gt 0 ]; then
        echo "🔄 Aguardando ${#ACTIVE_PIDS[@]} experimento(s)..."
        sleep 30
    fi
done

echo ""
echo "=========================================="
echo "🎉 Todos os experimentos concluídos!"
echo "📊 Execute 'julia scripts/analyze_compression_impact.jl' para analisar os resultados"
echo "📁 Resultados salvos em: results/optimized_retain_*.csv"
<parameter name="filePath">/home/ablima/Desktop/wave6g/scripts/run_optimized_experiments.sh