#!/bin/bash

# Script mestre para executar experimentos com paralelização avançada
# Interface unificada para todos os modos de execução

set -e  # Sair em caso de erro

# Configurações padrão
MODE=${1:-"optimized"}  # optimized, distributed, benchmark
NUM_GPUS=${2:-1}
MAX_CONCURRENT=${3:-2}

echo "🚀 WAVE6G - EXPERIMENTOS COM PARALELIZAÇÃO AVANÇADA"
echo "=================================================="
echo "Modo: $MODE"
echo "GPUs: $NUM_GPUS"
echo "Máximo simultâneo: $MAX_CONCURRENT"
echo ""

case $MODE in
    "optimized")
        echo "📊 MODO OTIMIZADO: Controle inteligente de recursos"
        echo "Características:"
        echo "  • Limitação de processos simultâneos"
        echo "  • Monitoramento de memória RAM/GPU"
        echo "  • Fila inteligente de experimentos"
        echo "  • Recuperação automática de falhas"
        echo "  • Paralelização CUDA com múltiplas streams"
        echo ""

        # Iniciar monitor em background
        bash scripts/monitor_experiments.sh &
        MONITOR_PID=$!

        # Executar experimentos otimizados
        MAX_CONCURRENT_EXPERIMENTS=$MAX_CONCURRENT bash scripts/run_optimized_experiments.sh

        # Parar monitor
        kill $MONITOR_PID 2>/dev/null || true
        ;;

    "cuda-test")
        echo "🎮 MODO CUDA TEST: Teste de paralelização GPU"
        echo "Características:"
        echo "  • Teste rápido de performance CUDA"
        echo "  • Comparação sequencial vs paralelo"
        echo "  • Otimização automática para RTX 2060"
        echo ""

        bash scripts/test_cuda_parallel.sh
        ;;

    "cuda-benchmark")
        echo "🏁 MODO CUDA BENCHMARK: Benchmark completo"
        echo "Características:"
        echo "  • Comparação CPU vs GPU"
        echo "  • Diferentes configurações de streams"
        echo "  • Análise detalhada de performance"
        echo ""

        bash scripts/benchmark_cuda_performance.sh
        ;;
    "distributed")
        echo "🌐 MODO DISTRIBUÍDO: Múltiplas GPUs"
        echo "Características:"
        echo "  • Balanceamento inteligente de carga"
        echo "  • Isolamento por GPU"
        echo "  • Monitoramento de memória GPU"
        echo "  • Consolidação automática de resultados"
        echo ""

        if [ $NUM_GPUS -gt 1 ]; then
            bash scripts/launch_distributed.sh $NUM_GPUS
        else
            echo "⚠️  Modo distribuído requer múltiplas GPUs. Usando modo single GPU."
            bash scripts/distributed_experiments.sh 0 1
        fi
        ;;

    "benchmark")
        echo "⚡ MODO BENCHMARK: Teste de performance"
        echo "Características:"
        echo "  • Medição de tempo para 1 janela"
        echo "  • Estimativa para cargas maiores"
        echo "  • Forçar uso de GPU"
        echo ""

        bash scripts/benchmark_experiment.sh
        ;;

    "parallel")
        echo "🔄 MODO PARALELO: Execução simultânea simples"
        echo "Características:"
        echo "  • Todos os experimentos simultâneos"
        echo "  • Sem controle de recursos"
        echo "  • Rápido para sistemas potentes"
        echo ""

        bash scripts/run_multiple_experiments.sh
        ;;

    *)
        echo "❌ Modo desconhecido: $MODE"
        echo ""
        echo "Modos disponíveis:"
        echo "  optimized      - Controle inteligente de recursos (recomendado)"
        echo "  distributed    - Múltiplas GPUs com balanceamento"
        echo "  benchmark      - Teste de performance rápido"
        echo "  parallel       - Execução simultânea simples"
        echo "  cuda-test      - Teste de paralelização CUDA"
        echo "  cuda-benchmark - Benchmark completo CPU vs GPU"
        echo ""
        echo "Uso: bash scripts/run_parallel_experiments.sh [modo] [num_gpus] [max_concurrent]"
        exit 1
        ;;
esac

echo ""
echo "🎉 Execução concluída!"
echo "📊 Para analisar resultados: julia scripts/analyze_compression_impact.jl"
echo "📁 Resultados em: results/"

# Análise automática se disponível
if [ -f "results/optimized_retain_0.01.csv" ] || [ -f "results/distributed_consolidated.csv" ]; then
    echo ""
    echo "🔄 Executando análise automática..."
    julia scripts/analyze_compression_impact.jl || echo "⚠️  Análise falhou - execute manualmente"
fi