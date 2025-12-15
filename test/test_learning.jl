# ======================================================================
# test/test_learning.jl
# ----------------------------------------------------------------------
# Validação do Motor de Aprendizado (Flux/AD)
# ======================================================================

using LinearAlgebra
using Statistics
using Random
using Printf

# NOTA: Assumimos que o módulo Wave6G já foi carregado no REPL antes de incluir este script
# NOTA 2: A função optimize_wavelet_sparsity deve ser acessível. 
# Se ela não for exportada, o script deve incluir a lógica do engine/learn_driver.jl
# Para simplicidade, vamos assumir que o fluxo de carregamento do Wave6G é suficiente.


function run_learning_validation()
    @info "Iniciando Teste de Validação do AD (Otimização de Esparsidade)"
    
    # 1. Geração de Dados de Teste
    # Usaremos um sinal senoidal ruidoso de 256 pontos (N=4*2^L com L=5) 
    # para garantir que haja alguma estrutura a ser aprendida.
    N_data = 256
    Random.seed!(101)
    
    t = range(0, 4π, length=N_data)
    # Sinal = Senóide + Ruído
    data = sin.(t) + 0.1 * randn(N_data) 
    
    # 2. Parâmetros de Otimização
    L_layers = 5            # 5 Camadas (256 -> 128 -> ... -> 8)
    chi_bond = 8
    chimid_bond = 6
    
    # Permitir ajuste de iterações via variável de ambiente para testes rápidos
    numiter = let
        # Prioridade: WAVE6G_TEST_ITERS > FAST_TESTS > default(200)
        if haskey(ENV, "WAVE6G_TEST_ITERS")
            try
                parse(Int, ENV["WAVE6G_TEST_ITERS"])
            catch
                200
            end
        elseif get(ENV, "FAST_TESTS", "0") in ("1", "true", "TRUE")
            30
        else
            200
        end
    end

    opts = Dict(
        "numiter" => numiter,
        "lr"      => 0.001  # Learning Rate do Adam
    )

    # 3. Chamada da API de Aprendizado
    # Esta função chama o motor `optimize_wavelet_sparsity` que usa o Flux.
    results = Wave6G.learn_wavelet_filters(
        data; 
        L=L_layers, 
        chi=chi_bond, 
        chimid=chimid_bond, 
        opts=opts
    )

    final_loss = results.loss
    
    @printf "\nRESULTADOS DA OTIMIZAÇÃO:\n"
    @printf "------------------------------------------------\n"
    @printf "Camadas MERA (L): %d\n" L_layers
    @printf "Iterações AD: %d\n" opts["numiter"]
    @printf "Loss Final (Esparsidade): %.6f\n" final_loss
    @printf "Iterações usadas: %d (FAST_TESTS=%s)\n" numiter get(ENV, "FAST_TESTS", "0")
    @printf "------------------------------------------------\n"
    
    # Critério de Sucesso: A Loss deve ter diminuído significativamente
    # (Ex: se a loss inicial for ~100.0, a final deve ser < 10.0)
    # Não podemos saber o valor inicial, mas esperamos que o valor final seja razoável.
    return final_loss < 10.0 # Critério arbitrário para sucesso
end


# Chame a função para execução
# run_learning_validation()
