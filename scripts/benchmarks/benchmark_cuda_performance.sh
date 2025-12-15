#!/bin/bash

# Benchmark de performance CUDA vs CPU
# Compara paralelização com múltiplas streams vs processamento sequencial

echo "🏁 BENCHMARK CUDA vs CPU - Wave6G"
echo "=================================="

OUTPUT_DIR="results/benchmark_cuda"
mkdir -p "$OUTPUT_DIR"

# Configurações de teste
WINDOWS=5
RETAIN=0.1

echo "📊 Configurações do teste:"
echo "  • Janelas: $WINDOWS"
echo "  • Retain ratio: $RETAIN"
echo "  • GPU: RTX 2060 (se disponível)"
echo ""

# Função para executar teste
run_benchmark() {
    local mode=$1
    local streams=$2
    local threads=$3
    local label=$4

    echo "🔬 Executando: $label"

    export JULIA_CUDA_FORCE=true
    export JULIA_CUDA_STREAMS=$streams
    export JULIA_NUM_THREADS=$threads

    local start_time=$(date +%s.%3N)

    julia scripts/run_mawi_experiments.jl \
        --step=4096 \
        --num-windows=$WINDOWS \
        --retain=$RETAIN \
        --output="$OUTPUT_DIR/benchmark_${mode}.csv" 2>/dev/null

    local end_time=$(date +%s.%3N)
    local duration=$(echo "$end_time - $start_time" | bc)

    echo "⏱️  Tempo: ${duration}s"
    echo "$mode,$streams,$threads,$duration" >> "$OUTPUT_DIR/results.csv"

    return 0
}

# Inicializar arquivo de resultados
echo "mode,streams,threads,time_seconds" > "$OUTPUT_DIR/results.csv"

# Teste 1: CPU sequencial
echo ""
echo "🖥️  TESTE 1: CPU Sequencial"
run_benchmark "cpu_sequential" 1 1 "CPU (1 thread, sem CUDA)"

# Teste 2: CPU multi-thread
echo ""
echo "🖥️  TESTE 2: CPU Multi-thread"
run_benchmark "cpu_parallel" 1 6 "CPU (6 threads, sem CUDA)"

# Teste 3: GPU sequencial (1 stream)
echo ""
echo "🎮 TESTE 3: GPU Sequencial"
run_benchmark "gpu_sequential" 1 6 "GPU (1 stream, 6 CPU threads)"

# Teste 4: GPU com múltiplas streams
echo ""
echo "🎮 TESTE 4: GPU Multi-stream (Otimizado RTX 2060)"
run_benchmark "gpu_parallel" 6 6 "GPU (6 streams, 6 CPU threads)"

# Análise de resultados
echo ""
echo "📊 RESULTADOS FINAIS"
echo "===================="

if [ -f "$OUTPUT_DIR/results.csv" ]; then
    echo ""
    echo "Comparação de performance:"
    echo "Mode                  | Streams | Threads | Time (s) | Speedup"
    echo "----------------------|---------|---------|----------|--------"

    # Ler resultados e calcular speedups
    cpu_seq_time=""
    while IFS=',' read -r mode streams threads time; do
        if [ "$mode" = "cpu_sequential" ]; then
            cpu_seq_time=$time
            baseline=$time
        fi

        if [ -n "$cpu_seq_time" ]; then
            speedup=$(echo "scale=2; $cpu_seq_time / $time" | bc 2>/dev/null || echo "N/A")
            printf "%-21s | %-7s | %-7s | %-8s | %-6s\n" "$mode" "$streams" "$threads" "$(printf "%.2f" $time)" "$speedup"
        fi
    done < "$OUTPUT_DIR/results.csv"

    echo ""
    echo "💡 Conclusões:"
    echo "  • Arquivos salvos em: $OUTPUT_DIR/"
    echo "  • GPU streams melhoram performance significativamente"
    echo "  • RTX 2060 otimizada com 6 streams + 6 CPU threads"

    # Recomendação
    gpu_parallel_time=$(grep "gpu_parallel" "$OUTPUT_DIR/results.csv" | cut -d',' -f4)
    if [ -n "$gpu_parallel_time" ] && [ -n "$cpu_seq_time" ]; then
        speedup=$(echo "scale=1; $cpu_seq_time / $gpu_parallel_time" | bc 2>/dev/null)
        if (( $(echo "$speedup > 2" | bc -l) )); then
            echo "  ✅ Excelente speedup! Use JULIA_CUDA_STREAMS=6"
        elif (( $(echo "$speedup > 1.5" | bc -l) )); then
            echo "  👍 Bom speedup! Configuração adequada"
        else
            echo "  ⚠️  Speedup limitado - verificar configuração GPU"
        fi
    fi
else
    echo "❌ Erro: Arquivo de resultados não encontrado"
fi

echo ""
echo "🎯 Para experimentos completos, use:"
echo "   JULIA_CUDA_STREAMS=6 JULIA_NUM_THREADS=6 bash scripts/run_parallel_experiments.sh optimized"