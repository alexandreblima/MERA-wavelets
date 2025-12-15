using Pkg
Pkg.activate(".")

using Wave6G

println("Testando MERA básico...")

# Teste simples de inicialização
try
    data = rand(1024)
    println("1. Dados criados: $(length(data)) pontos")

    state = Wave6G.prepare_variational_state(data; L=3, chi=4, chimid=4, normalize=true)
    println("2. Estado variacional preparado: L=$(state.L), χ=$(state.chi), χ_mid=$(state.chimid)")

    println("✅ MERA inicialização básica funcionou!")
catch e
    println("❌ Erro na inicialização MERA:")
    println("Tipo: $(typeof(e))")
    println("Mensagem: $(e)")
    for (i, frame) in enumerate(stacktrace())
        println("  [$i] $(frame)")
        i > 10 && break
    end
end
