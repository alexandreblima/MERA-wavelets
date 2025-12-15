#!/bin/bash

# Script para testar paralelização CUDA na RTX 2060
# Foca em maximizar o uso da GPU com múltiplas streams

echo "🎮 TESTE DE PARALELIZAÇÃO CUDA - RTX 2060"
echo "=========================================="

# Verificar GPU
if ! nvidia-smi --query-gpu=name --format=csv,noheader,nounits | grep -q "RTX 2060"; then
    echo "❌ RTX 2060 não detectada!"
    exit 1
fi

echo "✅ RTX 2060 detectada"

# Configurações otimizadas para RTX 2060
export JULIA_CUDA_FORCE=true
export JULIA_CUDA_STREAMS=6  # 6 streams para RTX 2060 (6GB VRAM)
export JULIA_NUM_THREADS=6   # 6 threads CPU para combinar com GPU

echo "🔧 Configurações:"
echo "  • CUDA Streams: $JULIA_CUDA_STREAMS"
echo "  • CPU Threads: $JULIA_NUM_THREADS"
echo "  • GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"
echo ""

# Teste rápido com 1 janela para verificar paralelização
echo "🧪 Executando teste rápido (1 janela, retain=0.1)..."

start_time=$(date +%s)

julia scripts/run_mawi_experiments.jl \
    --step=4096 \
    --num-windows=1 \
    --retain=0.1 \
    --output="results/cuda_parallel_test.csv"

end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo "⏱️  Tempo do teste: ${duration} segundos"

# Verificar uso da GPU durante o teste
echo ""
echo "📊 Estatísticas da GPU durante o teste:"
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total \
    --format=csv,noheader,nounits | tail -5

echo ""
echo "✅ Teste concluído! Arquivo: results/cuda_parallel_test.csv"

# Comparar com versão sequencial (sem streams)
echo ""
echo "🔄 Comparando com versão sequencial..."

export JULIA_CUDA_STREAMS=1

start_time_seq=$(date +%s)

julia scripts/run_mawi_experiments.jl \
    --step=4096 \
    --num-windows=1 \
    --retain=0.1 \
    --output="results/cuda_sequential_test.csv"

end_time_seq=$(date +%s)
duration_seq=$((end_time_seq - start_time_seq))

echo ""
echo "⏱️  Tempo sequencial: ${duration_seq} segundos"
echo "📊 Speedup: $(echo "scale=2; $duration_seq / $duration" | bc)x"

if (( $(echo "$duration_seq > $duration" | bc -l) )); then
    echo "✅ Paralelização CUDA funcionando! Speedup de $(echo "scale=2; $duration_seq / $duration" | bc)x"
else
    echo "⚠️  Sem speedup detectado - verificar configuração"
fi

echo ""
echo "🎯 Recomendação: Use JULIA_CUDA_STREAMS=6 para RTX 2060 em experimentos completos"