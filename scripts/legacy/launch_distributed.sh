#!/bin/bash

# Launcher para Distributed Computing - Inicia múltiplas GPUs simultaneamente
# Uso: bash scripts/launch_distributed.sh [num_gpus]

NUM_GPUS=${1:-$(nvidia-smi --list-gpus | wc -l)}  # Usar todas as GPUs por padrão

echo "🚀 LAUNCHER DISTRIBUTED COMPUTING"
echo "================================="
echo "Iniciando $NUM_GPUS GPUs simultaneamente"
echo ""

# Verificar se temos GPUs suficientes
TOTAL_GPUS_AVAILABLE=$(nvidia-smi --list-gpus | wc -l)
if [ $NUM_GPUS -gt $TOTAL_GPUS_AVAILABLE ]; then
    echo "❌ Solicitadas $NUM_GPUS GPUs, mas apenas $TOTAL_GPUS_AVAILABLE disponíveis"
    exit 1
fi

# Array para armazenar PIDs
declare -a PIDS=()

# Iniciar uma instância por GPU
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    echo "🔥 Iniciando GPU $gpu_id..."

    # Executar em background
    bash scripts/distributed_experiments.sh $gpu_id $NUM_GPUS $TOTAL_GPUS_AVAILABLE &
    PIDS[$gpu_id]=$!

    echo "📋 GPU $gpu_id - PID: ${PIDS[$gpu_id]}"
    sleep 2  # Pequena pausa para evitar conflitos de inicialização
done

echo ""
echo "⏳ Aguardando conclusão de todas as GPUs..."
echo ""

# Aguardar todas as GPUs terminarem
for gpu_id in "${!PIDS[@]}"; do
    pid=${PIDS[$gpu_id]}
    echo "Aguardando GPU $gpu_id (PID: $pid)..."
    wait $pid
    echo "✅ GPU $gpu_id concluída!"
done

echo ""
echo "🎉 DISTRIBUTED COMPUTING CONCLUÍDO!"
echo "📊 Todas as GPUs finalizaram seus experimentos"
echo "📁 Resultados salvos com prefixos: distributed_gpu[0-$((NUM_GPUS-1))]_"

# Consolidar resultados (opcional)
echo ""
echo "🔄 Consolidando resultados..."
if command -v python3 &> /dev/null; then
    python3 -c "
import pandas as pd
import glob
import os

# Encontrar todos os arquivos distributed
files = glob.glob('results/distributed_gpu*.csv')
if files:
    print(f'Encontrados {len(files)} arquivos de resultados')
    # Consolidar em um único arquivo
    dfs = []
    for file in files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
        except Exception as e:
            print(f'Erro ao ler {file}: {e}')

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_csv('results/distributed_consolidated.csv', index=False)
        print('✅ Resultados consolidados em: results/distributed_consolidated.csv')
else:
    print('Nenhum arquivo distributed encontrado')
"
else
    echo "⚠️  Python3 não encontrado - pule a consolidação automática"
    echo "📝 Execute manualmente: julia scripts/analyze_compression_impact.jl"
fi