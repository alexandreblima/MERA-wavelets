#!/bin/bash

# Script melhorado para Distributed Computing - Executar experimentos em múltiplas GPUs/máquinas
# Com balanceamento de carga inteligente e monitoramento de recursos

GPU_ID=${1:-0}  # GPU ID padrão 0
NUM_GPUS=${2:-1}  # Número de GPUs padrão 1
TOTAL_GPUS=${3:-$(nvidia-smi --list-gpus | wc -l)}  # Total de GPUs no sistema

echo "🚀 DISTRIBUTED COMPUTING OTIMIZADO - Wave6G Experiments"
echo "=========================================="
echo "GPU ID atual: $GPU_ID"
echo "Total GPUs no sistema: $TOTAL_GPUS"
echo "GPUs utilizadas: $NUM_GPUS"
echo ""

# Verificar se CUDA está disponível
if ! nvidia-smi -i $GPU_ID >/dev/null 2>&1; then
    echo "❌ GPU $GPU_ID não encontrada!"
    exit 1
fi

# Função para verificar memória GPU disponível
check_gpu_memory() {
    local gpu_id=$1
    local mem_used=$(nvidia-smi --id=$gpu_id --query-gpu=memory.used --format=csv,noheader,nounits)
    local mem_total=$(nvidia-smi --id=$gpu_id --query-gpu=memory.total --format=csv,noheader,nounits)
    local mem_free=$((mem_total - mem_used))
    echo $mem_free
}

# Função para estimar carga de trabalho por retain_ratio
estimate_workload() {
    local retain_ratio=$1
    # Retain ratios menores são mais custosos (mais coeficientes para processar)
    # Usar uma função exponencial inversa para estimar
    local workload=$(echo "scale=2; 1 / ($retain_ratio + 0.01)" | bc)
    echo $workload
}

# Configurar CUDA_VISIBLE_DEVICES para usar apenas a GPU especificada
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "🔧 Configurado CUDA_VISIBLE_DEVICES=$GPU_ID"

# Verificar memória disponível na GPU
GPU_MEMORY_MB=$(check_gpu_memory $GPU_ID)
echo "💾 Memória GPU disponível: ${GPU_MEMORY_MB}MB"

# Definir experimentos com base na capacidade da GPU
RETAIN_RATES=(0.01 0.02 0.05 0.1 0.2)
TOTAL_EXPERIMENTS=${#RETAIN_RATES[@]}

# Estratégia de balanceamento inteligente
if [ $NUM_GPUS -gt 1 ]; then
    echo "📊 Usando balanceamento inteligente de carga"

    # Calcular workload total e por GPU
    total_workload=0
    declare -a workloads=()
    for rate in "${RETAIN_RATES[@]}"; do
        workload=$(estimate_workload $rate)
        workloads+=($workload)
        total_workload=$(echo "$total_workload + $workload" | bc)
    done

    target_workload_per_gpu=$(echo "scale=2; $total_workload / $NUM_GPUS" | bc)
    echo "🎯 Carga total: $total_workload | Carga por GPU: $target_workload_per_gpu"

    # Atribuir experimentos usando algoritmo greedy
    assigned_experiments=()
    current_workload=0
    target=$target_workload_per_gpu

    for i in $(seq 0 $((TOTAL_EXPERIMENTS - 1))); do
        # Verificar se adicionar este experimento mantém balanceamento
        test_workload=$(echo "$current_workload + ${workloads[$i]}" | bc)

        if [ $GPU_ID -eq 0 ] || [ $(echo "$test_workload <= $target * 1.2" | bc -l) -eq 1 ]; then
            assigned_experiments+=($i)
            current_workload=$test_workload
        fi
    done

    echo "� GPU $GPU_ID recebeu ${#assigned_experiments[@]} experimento(s)"
else
    # Modo single GPU - executar todos os experimentos
    echo "📊 Modo Single GPU: Executando todos os experimentos"
    assigned_experiments=($(seq 0 $((TOTAL_EXPERIMENTS - 1))))
fi

# Executar experimentos atribuídos
for idx in "${assigned_experiments[@]}"; do
    rate=${RETAIN_RATES[$idx]}
    workload=${workloads[$idx]}

    echo ""
    echo "GPU $GPU_ID executando experimento retain_ratio = $rate (carga: $workload)"

    # Verificar memória antes de iniciar
    current_mem=$(check_gpu_memory $GPU_ID)
    if [ $current_mem -lt 1024 ]; then  # Menos de 1GB livre
        echo "⚠️  Memória GPU baixa (${current_mem}MB) - aguardando liberação..."
        sleep 60
    fi

    JULIA_NUM_THREADS=$(nproc) JULIA_CUDA_FORCE=true julia scripts/run_mawi_experiments.jl \
        --step=4096 \
        --num-windows=100 \
        --retain=$rate \
        --output="results/distributed_gpu${GPU_ID}_retain_${rate}.csv"

    echo "✅ GPU $GPU_ID concluiu retain_ratio = $rate"
done

echo ""
echo "✅ GPU $GPU_ID concluiu todos os seus experimentos!"
echo "📁 Resultados salvos com prefixo: distributed_gpu${GPU_ID}_"