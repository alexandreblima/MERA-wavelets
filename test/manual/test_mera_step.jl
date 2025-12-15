using Pkg
Pkg.activate(".")

include("engine/mera_utils.jl")
include("engine/learn_driver.jl")
include("engine/helpers_merawave.jl")
include("engine/doVarMERA.jl")
include("engine/doConformalMERA.jl")
include("engine/mainVarMERA.jl")

using Wave6G

println("Testando passo a passo a inicialização MERA...")

try
    data = rand(1024)
    println("1. Dados criados: $(length(data)) pontos")

    # Teste apenas a normalização
    data_f64 = Float64.(data)
    println("2. Conversão para Float64: OK")

    # Teste apenas a inicialização dos tensores
    L, chi, chimid = 3, 4, 4
    println("3. Testando _initial_tensors com L=$L, chi=$chi, chimid=$chimid")
    tensors = Wave6G._initial_tensors(L, chi, chimid, :random)
    println("4. Tensores inicializados: wC=$(length(tensors.wC)), vC=$(length(tensors.vC)), uC=$(length(tensors.uC))")

    println("✅ Inicialização dos tensores funcionou!")
catch e
    println("❌ Erro na inicialização:")
    println("Tipo: $(typeof(e))")
    println("Mensagem: $(e)")
    for (i, frame) in enumerate(stacktrace())
        println("  [$i] $(frame)")
        i > 10 && break
    end
end
