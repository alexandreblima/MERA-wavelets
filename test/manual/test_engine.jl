using Pkg
Pkg.activate(".")

println("Testando carregamento dos módulos engine...")

try
    # Teste apenas o carregamento dos módulos
    include("engine/mera_utils.jl")
    println("1. mera_utils.jl carregado")

    include("engine/learn_driver.jl")
    println("2. learn_driver.jl carregado")

    include("engine/helpers_merawave_clean.jl")
    println("3. helpers_merawave_clean.jl carregado")

    include("engine/doVarMERA.jl")
    println("4. doVarMERA.jl carregado")

    include("engine/doConformalMERA.jl")
    println("5. doConformalMERA.jl carregado")

    include("engine/mainVarMERA.jl")
    println("6. mainVarMERA.jl carregado")

    println("✅ Todos os módulos engine carregados com sucesso!")
catch e
    println("❌ Erro no carregamento dos módulos:")
    println("Tipo: $(typeof(e))")
    println("Mensagem: $(e)")
    for (i, frame) in enumerate(stacktrace())
        println("  [$i] $(frame)")
        i > 10 && break
    end
end
