using .Wave6G
using CUDA

# Dados de teste maiores
x = rand(Float32, 1024)  # Aumentei o tamanho dos dados

# Parâmetros para teste mais realista
L = 3
chi = 4
chimid = 4
opts = Dict("numiter" => 10, "lr" => 0.01)  # Mais iterações

# Chama a função de aprendizado
wC, vC, uC, final_loss = Wave6G.optimize_wavelet_sparsity(x; L=L, chi=chi, chimid=chimid, opts=opts)

println("Teste finalizado. Loss: ", final_loss)
println("Tempo estimado para 200 iterações: ~", (10 / 1) * 200, " segundos (baseado em 10 iterações)")
