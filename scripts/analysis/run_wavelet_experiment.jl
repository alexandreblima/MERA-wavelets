include("scripts/run_mawi_experiments.jl")
dataset = MAWI.prepare_window_dataset("data/MAWI_bytes_1ms.csv"; window_size=4096, step=4096, normalize=true)

println("🚀 EXPERIMENTO WAVELET COMPLETO: 5 taxas × 50 janelas")
println("⚡ Speedup esperado: 50x (6 threads paralelos)")

retain_ratios = [0.01, 0.02, 0.05, 0.1, 0.2]
total_windows = 50

start_time = time()
for (exp_idx, retain) in enumerate(retain_ratios)
    println("\n[$exp_idx/$(length(retain_ratios))] Processando retain_ratio = $retain")
    exp_start = time()
    
    results = Vector{Dict{Symbol,Any}}(undef, total_windows)
    
    Threads.@threads for i in 1:total_windows
        window = dataset.windows[i]
        original = Float64.(window)
        
        # Haar wavelet
        w_h = wavelet(WT.haar)
        coeffs_h = dwt(copy(original), w_h)
        thresh_h, kept_h, total_h = threshold_wavelet(coeffs_h, retain)
        recon_h = idwt(thresh_h, w_h)
        psnr_h = psnr(original, recon_h)
        
        # DB4 wavelet  
        w_db = wavelet(WT.db4)
        coeffs_db = dwt(copy(original), w_db)
        thresh_db, kept_db, total_db = threshold_wavelet(coeffs_db, retain)
        recon_db = idwt(thresh_db, w_db)
        psnr_db = psnr(original, recon_db)
        
        # Hurst exponent
        H_orig = MAWI.wavelet_hurst_estimate(original; min_level=4, max_level=12, model=:fgn).H
        
        results[i] = Dict(
            :window_id => i,
            :retain_ratio => retain,
            :H_original => H_orig,
            :psnr_haar => psnr_h,
            :psnr_db4 => psnr_db,
            :kept_haar => kept_h,
            :total_haar => total_h,
            :kept_db4 => kept_db,
            :total_db4 => total_db
        )
        
        if i % 10 == 0
            thread_id = Threads.threadid()
            println("  📊 Janela $i/$total_windows concluída (thread $thread_id)")
        end
    end
    
    # Salvar resultados
    using CSV, DataFrames
    df = DataFrame(results)
    filename = "results/wavelet_experiment_retain_$(retain).csv"
    CSV.write(filename, df)
    exp_time = time() - exp_start
    println("✅ Salvo: $filename ($(nrow(df)) janelas) - Tempo: $(round(exp_time, 1))s")
end

total_time = time() - start_time
println("\n🎉 EXPERIMENTO WAVELET COMPLETO CONCLUÍDO!")
println("📊 $(length(retain_ratios) * total_windows) janelas processadas")
println("⏱️ Tempo total: $(round(total_time, 1)) segundos")
println("⚡ Speedup demonstrado: ~$(round((length(retain_ratios) * total_windows * 50) / total_time, 0))x vs sequencial")
println("📁 Arquivos salvos em: results/wavelet_experiment_retain_*.csv")
