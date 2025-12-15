# ======================================================================
# run_learning_test.jl
# ----------------------------------------------------------------------
# Wrapper para execução em modo Batch (Bash)
# ======================================================================

# Carrega o módulo principal
using Wave6G 

# Inclui o script de teste que define run_learning_validation()
# Assume que test_learning.jl está em ./test/
include("test/test_learning.jl") 

# Executa a função principal de teste
success = run_learning_validation()

# Opcional: imprime o sucesso/falha no final
println("\n[FIM] Teste do Ciclo de Aprendizado (Success): ", success)
