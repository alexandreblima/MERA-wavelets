using Wave6G

include("test_learning.jl")
include("test_reconstruction.jl")
include("test_wavelet_aliases.jl")

success_learning = run_learning_validation()
success_reconstruction = run_reconstruction_test(L_test, N_test, chi_test, chimid_test)
success_wavelets = run_wavelet_alias_tests()

all_success = success_learning && success_reconstruction && success_wavelets

println("\nSUMMARY -> Learning: ", success_learning, ", Reconstruction: ", success_reconstruction, ", Wavelet aliases: ", success_wavelets)

if !all_success
    error("Tests failed: learning=" * string(success_learning) * ", reconstruction=" * string(success_reconstruction) * ", wavelet_aliases=" * string(success_wavelets))
end

println("All tests passed.")
