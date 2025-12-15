#!/bin/bash

# Teste focado na otimização MERA - o verdadeiro gargalo
# Compara paralelização CUDA vs sequencial no MERA optimization

echo "🎯 TESTE FOCADO: Otimização MERA com CUDA"
echo "=========================================="

# Verificar GPU
if ! nvidia-smi --query-gpu=name --format=csv,noheader,nounits | grep -q "RTX 2060"; then
    echo "❌ RTX 2060 não detectada!"
    exit 1
fi

echo "✅ RTX 2060 detectada"

# Configurações para teste focado no MERA
export JULIA_CUDA_FORCE=true
export JULIA_NUM_THREADS=1  # Apenas 1 thread para focar na GPU

echo "📊 Configurações:"
echo "  • CUDA Streams: Testando 1 vs 6"
echo "  • CPU Threads: 1 (foco na GPU)"
echo "  • MERA: L=5, chi=4 (maior para testar paralelismo)"
echo "  • Iterations: 30 (teste rápido)"
echo ""

# Criar script Julia focado no MERA
cat > /tmp/test_mera.jl << 'EOF'
#!/usr/bin/env julia

using Pkg
# Corrigir caminho do projeto
project_dir = get(ENV, "WAVE6G_PROJECT_DIR", "/home/ablima/Desktop/wave6g")
Pkg.activate(joinpath(project_dir))

using Wave6G
using CUDA
using ArgParse

function time_mera_optimization(streams::Int, iterations::Int=50)
    println("🔬 Testando com $streams streams CUDA, $iterations iterações")

    # Configurar streams usando Wave6G
    Wave6G._cuda_streams[] = [CUDA.CuStream() for _ in 1:streams]
    @info "Streams configurados: $streams"

    # Dados de teste (sinal sintético)
    data = randn(Float32, 1024)  # Sinal menor para teste rápido

    # Configuração MERA maior para testar paralelismo
    L, chi, chimid = 5, 4, 4  # MERA maior com mais níveis para paralelizar

    # Medir tempo
    start_time = time()

    # Executar otimização
    schedule = [(L=L, chi=chi, chimid=chimid, numiter=iterations, lr=0.01f0, reinit=true, init=:random)]
    base_opts = Dict{String,Any}("mse_weight"=>0.0f0, "sparsity_weight"=>1.0f0)

    state = prepare_variational_state(data; L=L, chi=chi, chimid=chimid, normalize=false, init=:random)
    result = optimize_variational_schedule!(state; schedule=schedule, base_opts=base_opts, data=data)

    end_time = time()
    duration = end_time - start_time

    println("⏱️  Tempo: $(round(duration, digits=2))s")
    println("📊 Loss final: $(round(result.loss, digits=6))")

    return duration, result.loss
end

# Argumentos
s = ArgParseSettings()
@add_arg_table s begin
    "--streams"
        help = "Número de streams CUDA"
        arg_type = Int
        default = 1
    "--iterations"
        help = "Número de iterações"
        arg_type = Int
        default = 50
end

args = parse_args(s)

duration, loss = time_mera_optimization(args["streams"], args["iterations"])
println("RESULTADO: streams=$(args["streams"]) time=$(round(duration, digits=2)) loss=$(round(loss, digits=6))")
EOF

echo "🧪 Executando teste sequencial (1 stream)..."
export WAVE6G_PROJECT_DIR="/home/ablima/Desktop/wave6g"
SEQUENTIAL_TIME=$(julia /tmp/test_mera.jl --streams 1 --iterations 30 | grep "RESULTADO:" | sed 's/.*time=\([0-9.]*\).*/\1/')

echo ""
echo "🧪 Executando teste paralelo (6 streams)..."
export WAVE6G_PROJECT_DIR="/home/ablima/Desktop/wave6g"
PARALLEL_TIME=$(julia /tmp/test_mera.jl --streams 6 --iterations 30 | grep "RESULTADO:" | sed 's/.*time=\([0-9.]*\).*/\1/')

echo ""
echo "📊 RESULTADOS FINAIS"
echo "===================="

if [ -n "$SEQUENTIAL_TIME" ] && [ -n "$PARALLEL_TIME" ]; then
    echo "Tempo sequencial (1 stream): ${SEQUENTIAL_TIME}s"
    echo "Tempo paralelo (6 streams): ${PARALLEL_TIME}s"

    SPEEDUP=$(echo "scale=2; $SEQUENTIAL_TIME / $PARALLEL_TIME" | bc -l 2>/dev/null || echo "N/A")
    if [ "$SPEEDUP" != "N/A" ]; then
        echo "🚀 Speedup: ${SPEEDUP}x"

        if (( $(echo "$SPEEDUP > 1.2" | bc -l) )); then
            echo "✅ Sucesso! Paralelização CUDA funcionando no MERA"
        elif (( $(echo "$SPEEDUP > 1.0" | bc -l) )); then
            echo "⚠️  Speedup modesto - considere otimizar mais"
        else
            echo "❌ Sem speedup - paralelização pode ter overhead"
        fi
    fi
else
    echo "❌ Erro ao obter tempos de execução"
fi

# Limpar
rm -f /tmp/test_mera.jl

echo ""
echo "💡 Recomendações:"
echo "  • Use JULIA_CUDA_STREAMS=6 para MERA optimization"
echo "  • Para wavelets, o speedup é limitado pois são rápidos"
echo "  • Foco na paralelização do MERA que é o gargalo"