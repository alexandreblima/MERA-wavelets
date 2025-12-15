#!/usr/bin/env julia#!/usr/bin/env julia



using Pkgusing Pkg

Pkg.activate(joinpath(@__DIR__, ".."))Pkg.activate(joinpath(@__DIR__, ".."))



using ArgParseusing ArgParse

using CSVusing CSV

using DataFramesusing DataFrames

using Statisticsusing Statistics

using Waveletsusing Wavelets

using Wave6Gusing Wave6G

include("../src/MAWI.jl")include("../src/MAWI.jl")

using .MAWIusing .MAWI

using CUDAusing CUDA



const USE_GPU = Ref(false)# Simple debug-friendly runner: sequential, small defaults to validate pipeline

const USE_GPU = Ref(false)

function parse_cli()

    s = ArgParseSettings()function parse_cli()

    @add_arg_table! s begin    s = ArgParseSettings()

        "--data"; arg_type=String; default="data/small_test.csv"    @add_arg_table! s begin

        "--window-size"; arg_type=Int; default=4096        "--data"; arg_type=String; default="data/small_test.csv"

        "--step"; arg_type=Int; default=4096        "--window-size"; arg_type=Int; default=4096

        "--retain"; arg_type=Float64; default=0.1        "--step"; arg_type=Int; default=4096

        "--start-window"; arg_type=Int; default=1        "--retain"; arg_type=Float64; default=0.1

        "--num-windows"; arg_type=Int; default=1        "--start-window"; arg_type=Int; default=1

        "--mera-L"; arg_type=Int; default=5        "--num-windows"; arg_type=Int; default=1

        "--mera-chi"; arg_type=Int; default=4        "--mera-L"; arg_type=Int; default=5

        "--mera-chimid"; arg_type=Int; default=4        "--mera-chi"; arg_type=Int; default=4

        "--stage1-iters"; arg_type=Int; default=10        "--mera-chimid"; arg_type=Int; default=4

        "--stage1-lr"; arg_type=Float64; default=5e-3        "--stage1-iters"; arg_type=Int; default=10

        "--stage2-iters"; arg_type=Int; default=10        "--stage1-lr"; arg_type=Float64; default=5e-3

        "--stage2-lr"; arg_type=Float64; default=2.5e-3        "--stage2-iters"; arg_type=Int; default=10

        "--output"; arg_type=String; default="results/debug_test.csv"        "--stage2-lr"; arg_type=Float64; default=2.5e-3

        "--use-gpu"; action=:store_true        "--output"; arg_type=String; default="results/debug_test.csv"

        "--warm-start-haar"; action=:store_true        "--use-gpu"; action=:store_true

    end        "--warm-start-haar"; action=:store_true

    return parse_args(s)    end

end    return parse_args(s)

end

psnr(original, reconstructed) = begin

    mse = mean(abs2, reconstructed .- original)psnr(original, reconstructed) = begin

    max_val = maximum(abs.(original))    mse = mean(abs2, reconstructed .- original)

    mse < eps() || max_val == 0 ? Inf : 10.0 * log10(max_val^2 / mse)    max_val = maximum(abs.(original))

end    mse < eps() || max_val == 0 ? Inf : 10.0 * log10(max_val^2 / mse)

end

to_device(x) = USE_GPU[] ? cu(x) : x

from_device(x) = x isa CUDA.AbstractGPUArray ? Array(x) : xto_device(x) = USE_GPU[] ? cu(x) : x

from_device(x) = x isa CUDA.AbstractGPUArray ? Array(x) : x

function run_window_optimized(window_id, signal, cfg)

    tryfunction run_window_optimized(window_id, signal, cfg)

        println("Starting window $window_id")    try

        original = to_device(Float64.(signal))        println("Starting window $window_id")

        schedule = [ (L=cfg["mera-L"], chi=cfg["mera-chi"], chimid=cfg["mera-chimid"], numiter=cfg["stage1-iters"], lr=cfg["stage1-lr"], reinit=true, init=(cfg["warm-start-haar"] ? :haar : :random)),        original = to_device(Float64.(signal))

                    (numiter=cfg["stage2-iters"], lr=cfg["stage2-lr"], reinit=false) ]        schedule = [ (L=cfg["mera-L"], chi=cfg["mera-chi"], chimid=cfg["mera-chimid"], numiter=cfg["stage1-iters"], lr=cfg["stage1-lr"], reinit=true, init=(cfg["warm-start-haar"] ? :haar : :random)),

        state = Wave6G.prepare_variational_state(original; L=cfg["mera-L"], chi=cfg["mera-chi"], chimid=cfg["mera-chimid"], normalize=true, init=(cfg["warm-start-haar"] ? :haar : :random))                    (numiter=cfg["stage2-iters"], lr=cfg["stage2-lr"], reinit=false) ]

        state = to_device(state)        state = Wave6G.prepare_variational_state(original; L=cfg["mera-L"], chi=cfg["mera-chi"], chimid=cfg["mera-chimid"], normalize=true, init=(cfg["warm-start-haar"] ? :haar : :random))

        state, loss_history = Wave6G.optimize_variational_schedule!(state, schedule)        state = to_device(state)

        coeffs_mera = Wave6G.mera_coeffs(state)        state, loss_history = Wave6G.optimize_variational_schedule!(state, schedule)

        filtered_mera, kept, total = Wave6G.threshold_coeffs(coeffs_mera, cfg["retain"])        coeffs_mera = Wave6G.mera_coeffs(state)

        reconstructed = Wave6G.mera_reconstruct(state, filtered_mera)        filtered_mera, kept, total = Wave6G.threshold_coeffs(coeffs_mera, cfg["retain"])

        p = psnr(from_device(original), from_device(reconstructed))        reconstructed = Wave6G.mera_reconstruct(state, filtered_mera)

        return Dict(:window_id=>window_id, :psnr=>p, :kept=>kept, :total=>total, :error=>"")        p = psnr(from_device(original), from_device(reconstructed))

    catch e        return Dict(:window_id=>window_id, :psnr=>p, :kept=>kept, :total=>total, :error=>"")

        @error "window error" window_id exception=(e, catch_backtrace())    catch e

        return Dict(:window_id=>window_id, :error=>string(e))        @error "window error" window_id exception=(e, catch_backtrace())

    end        return Dict(:window_id=>window_id, :error=>string(e))

end    end

end

function main()

    args = parse_cli()function main()

    if args["use-gpu"]    args = parse_cli()

        if CUDA.functional()    if args["use-gpu"]

            USE_GPU[] = true        if CUDA.functional()

            CUDA.allowscalar(false)            USE_GPU[] = true

            @info "GPU enabled"            CUDA.allowscalar(false)

        else            @info "GPU enabled"

            @warn "GPU requested but not functional. Using CPU."        else

            USE_GPU[] = false            @warn "GPU requested but not functional. Using CPU."

        end            USE_GPU[] = false

    end        end

    end

    dataset = MAWI.prepare_window_dataset(args["data"]; window_size=args["window-size"], step=args["step"], normalize=true)

    windows = dataset.windows    dataset = MAWI.prepare_window_dataset(args["data"]; window_size=args["window-size"], step=args["step"], normalize=true)

    start = max(1, args["start-window"])    windows = dataset.windows

    n = args["num-windows"] > 0 ? min(args["num-windows"], length(windows)-start+1) : length(windows)-start+1    start = max(1, args["start-window"])

    results = Vector{Dict{Symbol,Any}}(undef, n)    n = args["num-windows"] > 0 ? min(args["num-windows"], length(windows)-start+1) : length(windows)-start+1

    for i in 1:n    results = Vector{Dict{Symbol,Any}}(undef, n)

        idx = start + i - 1    for i in 1:n

        results[i] = run_window_optimized(idx, windows[idx], args)        idx = start + i - 1

    end        results[i] = run_window_optimized(idx, windows[idx], args)

    df = DataFrame(results)    end

    mkpath(dirname(args["output"]))    df = DataFrame(results)

    CSV.write(args["output"], df)    mkpath(dirname(args["output"]))

    @info "Wrote $(nrow(df)) rows to $(args["output"])"    CSV.write(args["output"], df)

end    @info "Wrote $(nrow(df)) rows to $(args["output"])"

end

main()

main()
#!/usr/bin/env julia#!/usr/bin/env julia#!/usr/bin/env julia#!/usr/bin/env julia#!/usr/bin/env julia



using Pkg

Pkg.activate(joinpath(@__DIR__, ".."))

using Pkg

using ArgParse

using CSVPkg.activate(joinpath(@__DIR__, ".."))

using DataFrames

using Statisticsusing Pkg

using Wavelets

using Wave6Gusing ArgParse

include("../src/MAWI.jl")

using .MAWIusing CSVPkg.activate(joinpath(@__DIR__, ".."))

using CUDA

using DataFrames

# Global flag for GPU usage

const USE_GPU = Ref(false)using Statisticsusing Pkgusing Pkg



function setup_gpu()using Wavelets

    if CUDA.functional()

        USE_GPU[] = trueusing Wave6Gusing ArgParse

        @info "CUDA is functional. Activating GPU support."

        CUDA.allowscalar(false)include("../src/MAWI.jl")

        num_streams = min(parse(Int, get(ENV, "JULIA_CUDA_STREAMS", "4")), 8)

        if !isassigned(Wave6G._cuda_streams) || length(Wave6G._cuda_streams[]) != num_streamsusing .MAWIusing CSVPkg.activate(joinpath(@__DIR__, ".."))Pk[ Info: Otimizações ativadas: BLAS Otimizado

            Wave6G._cuda_streams[] = [CUDA.CuStream() for _ in 1:num_streams]

        endusing CUDA

        @info "Using $num_streams CUDA streams for parallel operations."

        tryusing DataFrames

            CUDA.math_mode!(CUDA.FAST_MATH)

            @info "CUDA Math Mode set to FAST_MATH for performance."# Global flag for GPU usage

        catch e

            @warn "Could not set CUDA math mode: $e"const USE_GPU = Ref(false)using Statistics

        end

    else

        USE_GPU[] = false

        @warn "CUDA not functional. Falling back to CPU."function setup_gpu()using Wavelets

    end

end    if CUDA.functional()



# Disable mixed precision for debugging stability        USE_GPU[] = trueusing Wave6Gusing ArgParsecfg = parse_cli()

Wave6G.enable_mixed_precision!(false)

Wave6G.enable_blas_optimization!(true)        @info "CUDA is functional. Activating GPU support."

@info "Performance optimizations: Mixed Precision (Disabled for Debug) and Optimized BLAS."

        CUDA.allowscalar(false)include("../src/MAWI.jl")

struct ExperimentConfig

    data_path::String        num_streams = min(parse(Int, get(ENV, "JULIA_CUDA_STREAMS", "4")), 8)

    window_size::Int

    step::Int        if !isassigned(Wave6G._cuda_streams) || length(Wave6G._cuda_streams[]) != num_streamsusing .MAWIusing CSV@info "Preparando dataset" cfgvate(joinpath(@__DIR__, ".."))

    retain_ratio::Float64

    start_window::Int            Wave6G._cuda_streams[] = [CUDA.CuStream() for _ in 1:num_streams]

    num_windows::Int

    mera_L::Int        endusing CUDA

    mera_chi::Int

    mera_chimid::Int        @info "Using $num_streams CUDA streams for parallel operations."

    stage1_iters::Int

    stage1_lr::Float64        tryusing DataFrames

    stage2_iters::Int

    stage2_lr::Float64            CUDA.math_mode!(CUDA.FAST_MATH)

    output_path::String

    init_mode::Symbol            @info "CUDA Math Mode set to FAST_MATH for performance."# Global flag for GPU usage

end

        catch e

function parse_cli()

    s = ArgParseSettings(description="Run MERA optimization experiments on time series data.")            @warn "Could not set CUDA math mode: $e"const USE_GPU = Ref(false)using Statisticsusing ArgParse

    @add_arg_table! s begin

        "--data"        end

            help = "Path to the data CSV file."

            arg_type = String    else

            default = "data/small_test.csv"

        "--window-size"        USE_GPU[] = false

            help = "Size of each processing window."

            arg_type = Int        @warn "CUDA not functional. Falling back to CPU."function setup_gpu()using Waveletsusing CSV

            default = 4096

        "--step"    end

            help = "Step size between consecutive windows."

            arg_type = Intend    if CUDA.functional()

            default = 4096

        "--retain"

            help = "Coefficient retention ratio for thresholding."

            arg_type = Float64# Disable mixed precision for debugging stability        USE_GPU[] = trueusing Wave6Gusing DataFrames

            default = 0.1

        "--start-window"Wave6G.enable_mixed_precision!(false)

            help = "Index of the first window to process."

            arg_type = IntWave6G.enable_blas_optimization!(true)        @info "CUDA is functional. Activating GPU support."

            default = 1

        "--num-windows"@info "Performance optimizations: Mixed Precision (Disabled for Debug) and Optimized BLAS."

            help = "Total number of windows to process. 0 means all."

            arg_type = Int        CUDA.allowscalar(false)include("../src/MAWI.jl")using Statistics

            default = 1

        "--mera-L"struct ExperimentConfig

            help = "Number of levels in the MERA network."

            arg_type = Int    data_path::String        # Configure streams for parallel execution on the GPU

            default = 5

        "--mera-chi"    window_size::Int

            help = "Bond dimension of the MERA network."

            arg_type = Int    step::Int        num_streams = min(parse(Int, get(ENV, "JULIA_CUDA_STREAMS", "4")), 8)using .MAWIusing Wavelets

            default = 4

        "--mera-chimid"    retain_ratio::Float64

            help = "Intermediate bond dimension for MERA."

            arg_type = Int    start_window::Int        if !isassigned(Wave6G._cuda_streams) || length(Wave6G._cuda_streams[]) != num_streams

            default = 4

        "--stage1-iters"    num_windows::Int

            help = "Number of iterations for optimization stage 1."

            arg_type = Int    mera_L::Int            Wave6G._cuda_streams[] = [CUDA.CuStream() for _ in 1:num_streams]using CUDAusing Wave6G

            default = 10

        "--stage1-lr"    mera_chi::Int

            help = "Learning rate for optimization stage 1."

            arg_type = Float64    mera_chimid::Int        end

            default = 5e-3

        "--stage2-iters"    stage1_iters::Int

            help = "Number of iterations for optimization stage 2."

            arg_type = Int    stage1_lr::Float64        @info "Using $num_streams CUDA streams for parallel operations."include("../src/MAWI.jl")

            default = 10

        "--stage2-lr"    stage2_iters::Int

            help = "Learning rate for optimization stage 2."

            arg_type = Float64    stage2_lr::Float64        try

            default = 2.5e-3

        "--output"    output_path::String

            help = "Path for the output CSV file."

            arg_type = String    init_mode::Symbol            CUDA.math_mode!(CUDA.FAST_MATH)# Global flag for GPU usageusing .MAWI

            default = "results/debug_output.csv"

        "--warm-start-haar"end

            help = "Initialize MERA with Haar wavelet filters."

            action = :store_true            @info "CUDA Math Mode set to FAST_MATH for performance."

        "--use-gpu"

            help = "Enable GPU acceleration."function parse_cli()

            action = :store_true

    end    s = ArgParseSettings(description="Run MERA optimization experiments on time series data.")        catch econst USE_GPU = Ref(false)using CUDA

    parsed = parse_args(s)

        @add_arg_table! s begin

    if parsed["use-gpu"]

        setup_gpu()        "--data"            @warn "Could not set CUDA math mode: $e"

    else

        USE_GPU[] = false            help = "Path to the data CSV file."

        @info "GPU usage is disabled by command-line argument."

    end            arg_type = String        end



    init_mode = parsed["warm-start-haar"] ? :haar : :random            default = "data/small_test.csv"

    return ExperimentConfig(

        parsed["data"],        "--window-size"    else

        parsed["window-size"],

        parsed["step"],            help = "Size of each processing window."

        parsed["retain"],

        max(parsed["start-window"], 1),            arg_type = Int        USE_GPU[] = falsefunction setup_gpu()# Forçar uso do CUDA se disponível

        parsed["num-windows"],

        parsed["mera-L"],            default = 4096

        parsed["mera-chi"],

        parsed["mera-chimid"],        "--step"        @warn "CUDA not functional. Falling back to CPU."

        parsed["stage1-iters"],

        parsed["stage1-lr"],            help = "Step size between consecutive windows."

        parsed["stage2-iters"],

        parsed["stage2-lr"],            arg_type = Int    end    if CUDA.functional()if CUDA.functional() || haskey(ENV, "JULIA_CUDA_FORCE")

        parsed["output"],

        init_mode            default = 4096

    )

end        "--retain"end



psnr(original, reconstructed) = begin            help = "Coefficient retention ratio for thresholding."

    mse = mean(abs2, reconstructed .- original)

    max_val = maximum(abs.(original))            arg_type = Float64        USE_GPU[] = true    @info "CUDA detectado e funcional (ou forçado) - usando GPU"

    mse < eps() || max_val == 0 ? Inf : 10.0 * log10(max_val^2 / mse)

end            default = 0.1



to_device(data) = USE_GPU[] ? cu(data) : data        "--start-window"# Enable performance optimizations

from_device(data) = data isa CUDA.AbstractGPUArray ? Array(data) : data

            help = "Index of the first window to process."

function run_window_optimized(window_id::Int, start_idx::Int, signal::AbstractVector{<:Number}, cfg::ExperimentConfig)

    try            arg_type = IntWave6G.enable_mixed_precision!(false) # Desativado para debug        @info "CUDA is functional. Activating GPU support."    CUDA.allowscalar(false)  # Desabilitar operações escalares na GPU para performance

        println("--- [Thread $(Threads.threadid())] Starting Window $window_id ---")

        original = to_device(Float64.(signal))            default = 1

        

        schedule = [        "--num-windows"Wave6G.enable_blas_optimization!(true)

            (L=cfg.mera_L, chi=cfg.mera_chi, chimid=cfg.mera_chimid, numiter=cfg.stage1_iters, lr=cfg.stage1_lr, reinit=true, init=cfg.init_mode),

            (numiter=cfg.stage2_iters, lr=cfg.stage2_lr, reinit=false)            help = "Total number of windows to process. 0 means all."

        ]

            arg_type = Int@info "Performance optimizations: Mixed Precision (Disabled for Debug) and Optimized BLAS."        CUDA.allowscalar(false)

        mera_time = @elapsed begin

            println("[T$(Threads.threadid()) W$window_id] Preparing variational state...")            default = 1

            state = Wave6G.prepare_variational_state(original; L=cfg.mera_L, chi=cfg.mera_chi, chimid=cfg.mera_chimid, normalize=true, init=cfg.init_mode)

            state = to_device(state)        "--mera-L"

            println("[T$(Threads.threadid()) W$window_id] Optimizing MERA...")

            state, loss_history = Wave6G.optimize_variational_schedule!(state, schedule)            help = "Number of levels in the MERA network."

            println("[T$(Threads.threadid()) W$window_id] Extracting coefficients...")

            coeffs_mera = Wave6G.mera_coeffs(state)            arg_type = Intstruct ExperimentConfig        # Configure streams for parallel execution on the GPU    # Configurar múltiplas streams para paralelização

            println("[T$(Threads.threadid()) W$window_id] Thresholding coefficients...")

            filtered_mera, kept_mera, total_mera = Wave6G.threshold_coeffs(coeffs_mera, cfg.retain_ratio)            default = 5

            println("[T$(Threads.threadid()) W$window_id] Reconstructing signal...")

            reconstructed_mera = Wave6G.mera_reconstruct(state, filtered_mera)        "--mera-chi"    data_path::String

        end

                    help = "Bond dimension of the MERA network."

        psnr_mera = psnr(from_device(original), from_device(reconstructed_mera))

        println("--- [Thread $(Threads.threadid())] Finished Window $window_id. PSNR: $psnr_mera, Time: $mera_time ---")            arg_type = Int    window_size::Int        num_streams = min(parse(Int, get(ENV, "JULIA_CUDA_STREAMS", "4")), 8)    num_streams = min(parse(Int, get(ENV, "JULIA_CUDA_STREAMS", "4")), 8)  # Até 8 streams

        

        return Dict(            default = 4

            :window_id => window_id,

            :psnr_mera => psnr_mera,        "--mera-chimid"    step::Int

            :time_mera => mera_time,

            :kept_mera => kept_mera,            help = "Intermediate bond dimension for MERA."

            :total_mera => total_mera,

            :error => ""            arg_type = Int    retain_ratio::Float64        if !isassigned(Wave6G._cuda_streams) || length(Wave6G._cuda_streams[]) != num_streams    Wave6G._cuda_streams[] = [CUDA.CuStream() for _ in 1:num_streams]

        )

    catch e            default = 4

        @error "Error processing window" window_id start_idx exception=(e, catch_backtrace())

        return Dict(:window_id => window_id, :error => string(e))        "--stage1-iters"    start_window::Int

    end

end            help = "Number of iterations for optimization stage 1."



function ensure_output_dir(path::String)            arg_type = Int    num_windows::Int            Wave6G._cuda_streams[] = [CUDA.CuStream() for _ in 1:num_streams]    @info "CUDA configurado com $num_streams streams para paralelização"

    dir = dirname(path)

    isempty(dir) || dir == "." || isdir(dir) || mkpath(dir)            default = 10

end

        "--stage1-lr"    mera_L::Int

function main()

    cfg = parse_cli()            help = "Learning rate for optimization stage 1."

    @info "Configuration loaded. Preparing dataset..." cfg

    dataset = MAWI.prepare_window_dataset(cfg.data_path; window_size=cfg.window_size, step=cfg.step, normalize=true)            arg_type = Float64    mera_chi::Int        end

    

    all_windows = dataset.windows            default = 5e-3

    total_available = length(all_windows)

    start_idx = min(cfg.start_window, total_available)        "--stage2-iters"    mera_chimid::Int

    

    end_idx = cfg.num_windows > 0 ? min(start_idx + cfg.num_windows - 1, total_available) : total_available            help = "Number of iterations for optimization stage 2."

    windows_to_process = all_windows[start_idx:end_idx]

    num_windows_to_process = length(windows_to_process)            arg_type = Int    stage1_iters::Int        @info "Using $num_streams CUDA streams for parallel operations."    # Forçar inicialização do CUDA



    if num_windows_to_process == 0            default = 10

        @info "No windows to process. Exiting."

        return        "--stage2-lr"    stage1_lr::Float64

    end

            help = "Learning rate for optimization stage 2."

    @info "Processing $num_windows_to_process windows from index $start_idx to $end_idx."

    results = Vector{Dict{Symbol,Any}}(undef, num_windows_to_process)            arg_type = Float64    stage2_iters::Int        try    CUDA.device()

    

    # Use a sequential loop for debugging            default = 2.5e-3

    for i in 1:num_windows_to_process

        window_data = windows_to_process[i]        "--output"    stage2_lr::Float64

        global_id = start_idx + i - 1

        signal_start = 1 + (global_id - 1) * cfg.step            help = "Path for the output CSV file."

        

        results[i] = run_window_optimized(global_id, signal_start, window_data, cfg)            arg_type = String    output_path::String            CUDA.math_mode!(CUDA.FAST_MATH)else

    end

            default = "results/debug_output.csv"

    df = DataFrame(filter(d -> !haskey(d, :error) || d[:error] == "", results))

    if !empty(df)        "--warm-start-haar"    init_mode::Symbol

        ensure_output_dir(cfg.output_path)

        CSV.write(cfg.output_path, df)            help = "Initialize MERA with Haar wavelet filters."

        @info "Successfully processed $(nrow(df)) windows. Results saved to $(cfg.output_path)."

    else            action = :store_trueend            @info "CUDA Math Mode set to FAST_MATH for performance."    @warn "CUDA não funcional - usando CPU"

        @warn "No windows were processed successfully."

    end        "--use-gpu"

end

            help = "Enable GPU acceleration."

main()

            action = :store_true

    endfunction parse_cli()        catch eend

    parsed = parse_args(s)

        s = ArgParseSettings(description="Run MERA optimization experiments on time series data.")

    if parsed["use-gpu"]

        setup_gpu()    @add_arg_table! s begin            @warn "Could not set CUDA math mode: $e"

    else

        USE_GPU[] = false        "--data"

        @info "GPU usage is disabled by command-line argument."

    end            help = "Path to the data CSV file."        end# Ativar otimizações avançadas



    init_mode = parsed["warm-start-haar"] ? :haar : :random            arg_type = String

    return ExperimentConfig(

        parsed["data"],            default = "data/small_test.csv" # Default to small test file    elseWave6G.enable_mixed_precision!(true)

        parsed["window-size"],

        parsed["step"],        "--window-size"

        parsed["retain"],

        max(parsed["start-window"], 1),            help = "Size of each processing window."        USE_GPU[] = falseWave6G.enable_blas_optimization!(true)

        parsed["num-windows"],

        parsed["mera-L"],            arg_type = Int

        parsed["mera-chi"],

        parsed["mera-chimid"],            default = 4096        @warn "CUDA not functional. Falling back to CPU."@info "Otimizações ativadas: Precision Mista e BLAS Otimizado"

        parsed["stage1-iters"],

        parsed["stage1-lr"],        "--step"

        parsed["stage2-iters"],

        parsed["stage2-lr"],            help = "Step size between consecutive windows."    end

        parsed["output"],

        init_mode            arg_type = Int

    )

end            default = 4096endprintln("Before parse_cli")



psnr(original, reconstructed) = begin        "--retain"

    mse = mean(abs2, reconstructed .- original)

    max_val = maximum(abs.(original))            help = "Coefficient retention ratio for thresholding."cfg = parse_cli()

    mse < eps() || max_val == 0 ? Inf : 10.0 * log10(max_val^2 / mse)

end            arg_type = Float64



to_device(data) = USE_GPU[] ? cu(data) : data            default = 0.1# Enable performance optimizationsprintln("After parse_cli")

from_device(data) = data isa CUDA.AbstractGPUArray ? Array(data) : data

        "--start-window"

function run_window_optimized(window_id::Int, start_idx::Int, signal::AbstractVector{<:Number}, cfg::ExperimentConfig)

    try            help = "Index of the first window to process."Wave6G.enable_mixed_precision!(true)@info "Preparando dataset" cfg

        println("--- [Thread $(Threads.threadid())] Starting Window $window_id ---")

        original = to_device(Float64.(signal))            arg_type = Int

        

        schedule = [            default = 1Wave6G.enable_blas_optimization!(true)dataset = MAWI.prepare_window_dataset(cfg.data_path; window_size=cfg.window_size,

            (L=cfg.mera_L, chi=cfg.mera_chi, chimid=cfg.mera_chimid, numiter=cfg.stage1_iters, lr=cfg.stage1_lr, reinit=true, init=cfg.init_mode),

            (numiter=cfg.stage2_iters, lr=cfg.stage2_lr, reinit=false)        "--num-windows"

        ]

            help = "Total number of windows to process. 0 means all."@info "Performance optimizations enabled: Mixed Precision and Optimized BLAS."                                     step=cfg.step, normalize=true)

        mera_time = @elapsed begin

            println("[T$(Threads.threadid()) W$window_id] Preparing variational state...")            arg_type = Int

            state = Wave6G.prepare_variational_state(original; L=cfg.mera_L, chi=cfg.mera_chi, chimid=cfg.mera_chimid, normalize=true, init=cfg.init_mode)

            state = to_device(state)            default = 1 # Default to 1 window for debug    all_windows = dataset.windows

            println("[T$(Threads.threadid()) W$window_id] Optimizing MERA...")

            state, loss_history = Wave6G.optimize_variational_schedule!(state, schedule)        "--mera-L"

            println("[T$(Threads.threadid()) W$window_id] Extracting coefficients...")

            coeffs_mera = Wave6G.mera_coeffs(state)            help = "Number of levels in the MERA network."struct ExperimentConfig    total_available = length(all_windows)

            println("[T$(Threads.threadid()) W$window_id] Thresholding coefficients...")

            filtered_mera, kept_mera, total_mera = Wave6G.threshold_coeffs(coeffs_mera, cfg.retain_ratio)            arg_type = Int

            println("[T$(Threads.threadid()) W$window_id] Reconstructing signal...")

            reconstructed_mera = Wave6G.mera_reconstruct(state, filtered_mera)            default = 5    data_path::String    start_idx = min(cfg.start_window, total_available)

        end

                "--mera-chi"

        psnr_mera = psnr(from_device(original), from_device(reconstructed_mera))

        println("--- [Thread $(Threads.threadid())] Finished Window $window_id. PSNR: $psnr_mera, Time: $mera_time ---")            help = "Bond dimension of the MERA network."    window_size::Int    windows = all_windows[start_idx:end]

        

        return Dict(            arg_type = Int

            :window_id => window_id,

            :psnr_mera => psnr_mera,            default = 4    step::Int    total_windows = length(windows)

            :time_mera => mera_time,

            :kept_mera => kept_mera,        "--mera-chimid"

            :total_mera => total_mera,

            :error => ""            help = "Intermediate bond dimension for MERA."    retain_ratio::Float64    if cfg.num_windows > 0

        )

    catch e            arg_type = Int

        @error "Error processing window" window_id start_idx exception=(e, catch_backtrace())

        return Dict(:window_id => window_id, :error => string(e))            default = 4    start_window::Int        total_windows = min(total_windows, cfg.num_windows)

    end

end        "--stage1-iters"



function ensure_output_dir(path::String)            help = "Number of iterations for optimization stage 1."    num_windows::Int        windows = windows[1:total_windows]

    dir = dirname(path)

    isempty(dir) || dir == "." || isdir(dir) || mkpath(dir)            arg_type = Int

end

            default = 10 # Reduced for debug    mera_L::Int    end

function main()

    cfg = parse_cli()        "--stage1-lr"

    @info "Configuration loaded. Preparing dataset..." cfg

    dataset = MAWI.prepare_window_dataset(cfg.data_path; window_size=cfg.window_size, step=cfg.step, normalize=true)            help = "Learning rate for optimization stage 1."    mera_chi::Int

    

    all_windows = dataset.windows            arg_type = Float64

    total_available = length(all_windows)

    start_idx = min(cfg.start_window, total_available)            default = 5e-3    mera_chimid::Int    schedule = schedule_from_config(cfg)

    

    end_idx = cfg.num_windows > 0 ? min(start_idx + cfg.num_windows - 1, total_available) : total_available        "--stage2-iters"

    windows_to_process = all_windows[start_idx:end_idx]

    num_windows_to_process = length(windows_to_process)            help = "Number of iterations for optimization stage 2."    stage1_iters::Int    rows = Vector{Dict{Symbol,Any}}(undef, total_windows)



    if num_windows_to_process == 0            arg_type = Int

        @info "No windows to process. Exiting."

        return            default = 10 # Reduced for debug    stage1_lr::Float64

    end

        "--stage2-lr"

    @info "Processing $num_windows_to_process windows from index $start_idx to $end_idx."

    results = Vector{Dict{Symbol,Any}}(undef, num_windows_to_process)            help = "Learning rate for optimization stage 2."    stage2_iters::Int    # Otimização de paralelização baseada em recursos disponíveis

    

    # Use a sequential loop for debugging            arg_type = Float64

    for i in 1:num_windows_to_process

        window_data = windows_to_process[i]            default = 2.5e-3    stage2_lr::Float64    num_threads = Threads.nthreads()

        global_id = start_idx + i - 1

        signal_start = 1 + (global_id - 1) * cfg.step        "--output"

        

        results[i] = run_window_optimized(global_id, signal_start, window_data, cfg)            help = "Path for the output CSV file."    output_path::String    num_cores = Sys.CPU_THREADS

    end

            arg_type = String

    df = DataFrame(filter(d -> !haskey(d, :error) || d[:error] == "", results))

    if !empty(df)            default = "results/debug_output.csv"    init_mode::Symbol

        ensure_output_dir(cfg.output_path)

        CSV.write(cfg.output_path, df)        "--warm-start-haar"

        @info "Successfully processed $(nrow(df)) windows. Results saved to $(cfg.output_path)."

    else            help = "Initialize MERA with Haar wavelet filters."end    # Ajustar número de threads baseado na memória disponível

        @warn "No windows were processed successfully."

    end            action = :store_true

end

        "--use-gpu"    if CUDA.functional()

main()

            help = "Enable GPU acceleration."

            action = :store_truefunction parse_cli()        # Com GPU, usar menos threads para evitar contenção de memória

    end

    parsed = parse_args(s)    s = ArgParseSettings(description="Run MERA optimization experiments on time series data.")        effective_threads = min(num_threads, max(1, num_cores ÷ 2))

    

    if parsed["use-gpu"]    @add_arg_table! s begin        @info "GPU detectada - ajustando para $effective_threads threads efetivos"

        setup_gpu()

    else        "--data"    else

        USE_GPU[] = false

        @info "GPU usage is disabled by command-line argument."            help = "Path to the data CSV file."        # Sem GPU, usar todos os threads disponíveis

    end

            arg_type = String        effective_threads = num_threads

    init_mode = parsed["warm-start-haar"] ? :haar : :random

    return ExperimentConfig(            default = "data/MAWI_bytes_1ms.csv"    end

        parsed["data"],

        parsed["window-size"],        "--window-size"

        parsed["step"],

        parsed["retain"],            help = "Size of each processing window."    # Verificar memória disponível por thread

        max(parsed["start-window"], 1),

        parsed["num-windows"],            arg_type = Int    memory_per_thread_gb = Sys.total_memory() / (1024^3 * effective_threads)

        parsed["mera-L"],

        parsed["mera-chi"],            default = 4096    if memory_per_thread_gb < 0.5

        parsed["mera-chimid"],

        parsed["stage1-iters"],        "--step"        effective_threads = max(1, effective_threads ÷ 2)

        parsed["stage1-lr"],

        parsed["stage2-iters"],            help = "Step size between consecutive windows."        @warn "Memória baixa detectada - reduzindo para $effective_threads threads"

        parsed["stage2-lr"],

        parsed["output"],            arg_type = Int    end

        init_mode

    )            default = 4096

end

        "--retain"    @info "Iniciando processamento paralelo de $total_windows janelas" threads=effective_threads memory_per_thread="$(round(memory_per_thread_gb, digits=2))GB"

psnr(original, reconstructed) = begin

    mse = mean(abs2, reconstructed .- original)            help = "Coefficient retention ratio for thresholding."

    max_val = maximum(abs.(original))

    mse < eps() || max_val == 0 ? Inf : 10.0 * log10(max_val^2 / mse)            arg_type = Float64    println("Starting threads, total_windows=$total_windows, length(windows)=$(length(windows))")

end

            default = 0.1

to_device(data) = USE_GPU[] ? cu(data) : data

from_device(data) = data isa CUDA.AbstractGPUArray ? Array(data) : data        "--start-window"    # Processamento paralelo com controle de recursos



function run_window_optimized(window_id::Int, start_idx::Int, signal::AbstractVector{<:Number}, cfg::ExperimentConfig)            help = "Index of the first window to process."    processed = Threads.Atomic{Int}(0)

    try

        println("--- [Thread $(Threads.threadid())] Starting Window $window_id ---")            arg_type = Int    errors = Threads.Atomic{Int}(0)

        original = to_device(Float64.(signal))

                    default = 1

        schedule = [

            (L=cfg.mera_L, chi=cfg.mera_chi, chimid=cfg.mera_chimid, numiter=cfg.stage1_iters, lr=cfg.stage1_lr, reinit=true, init=cfg.init_mode),        "--num-windows"    Threads.@threads for i in 1:total_windows

            (numiter=cfg.stage2_iters, lr=cfg.stage2_lr, reinit=false)

        ]            help = "Total number of windows to process. 0 means all."        try



        mera_time = @elapsed begin            arg_type = Int            window = windows[i]

            println("[T$(Threads.threadid()) W$window_id] Preparing variational state...")

            state = Wave6G.prepare_variational_state(original; L=cfg.mera_L, chi=cfg.mera_chi, chimid=cfg.mera_chimid, normalize=true, init=cfg.init_mode)            default = 0            global_id = start_idx + i - 1

            state = to_device(state)

            println("[T$(Threads.threadid()) W$window_id] Optimizing MERA...")        "--mera-L"            signal_start = 1 + (global_id - 1) * cfg.step

            state, loss_history = Wave6G.optimize_variational_schedule!(state, schedule)

            println("[T$(Threads.threadid()) W$window_id] Extracting coefficients...")            help = "Number of levels in the MERA network."

            coeffs_mera = Wave6G.mera_coeffs(state)

            println("[T$(Threads.threadid()) W$window_id] Thresholding coefficients...")            arg_type = Int            thread_id = Threads.threadid()

            filtered_mera, kept_mera, total_mera = Wave6G.threshold_coeffs(coeffs_mera, cfg.retain_ratio)

            println("[T$(Threads.threadid()) W$window_id] Reconstructing signal...")            default = 5            current_processed = Threads.atomic_add!(processed, 1)

            reconstructed_mera = Wave6G.mera_reconstruct(state, filtered_mera)

        end        "--mera-chi"

        

        psnr_mera = psnr(from_device(original), from_device(reconstructed_mera))            help = "Bond dimension of the MERA network."            @info "Processando janela $current_processed/$total_windows" window=global_id start=signal_start thread=thread_id

        println("--- [Thread $(Threads.threadid())] Finished Window $window_id. PSNR: $psnr_mera, Time: $mera_time ---")

                    arg_type = Int

        return Dict(

            :window_id => window_id,            default = 4            result = run_window_optimized(global_id, signal_start, window, cfg, schedule)

            :psnr_mera => psnr_mera,

            :time_mera => mera_time,        "--mera-chimid"            @info "Janela processada com sucesso" window=global_id result_keys=keys(result)

            :kept_mera => kept_mera,

            :total_mera => total_mera,            help = "Intermediate bond dimension for MERA."

            :error => ""

        )            arg_type = Int            rows[i] = result

    catch e

        @error "Error processing window" window_id start_idx exception=(e, catch_backtrace())            default = 4

        return Dict(:window_id => window_id, :error => string(e))

    end        "--stage1-iters"            # Pequena pausa para reduzir contenção de recursos

end

            help = "Number of iterations for optimization stage 1."            if current_processed % 10 == 0

function ensure_output_dir(path::String)

    dir = dirname(path)            arg_type = Int                sleep(0.01)

    isempty(dir) || dir == "." || isdir(dir) || mkpath(dir)

end            default = 300            end



function main()        "--stage1-lr"

    cfg = parse_cli()

    @info "Configuration loaded. Preparing dataset..." cfg            help = "Learning rate for optimization stage 1."        catch e

    dataset = MAWI.prepare_window_dataset(cfg.data_path; window_size=cfg.window_size, step=cfg.step, normalize=true)

                arg_type = Float64            Threads.atomic_add!(errors, 1)

    all_windows = dataset.windows

    total_available = length(all_windows)            default = 5e-3            @error "Erro na janela $i" exception=e

    start_idx = min(cfg.start_window, total_available)

            "--stage2-iters"            # Continuar processamento mesmo com erros

    end_idx = cfg.num_windows > 0 ? min(start_idx + cfg.num_windows - 1, total_available) : total_available

    windows_to_process = all_windows[start_idx:end_idx]            help = "Number of iterations for optimization stage 2."            rows[i] = Dict(:window_id => i, :error => string(e))

    num_windows_to_process = length(windows_to_process)

            arg_type = Int        end

    if num_windows_to_process == 0

        @info "No windows to process. Exiting."            default = 200    end

        return

    end        "--stage2-lr"



    @info "Processing $num_windows_to_process windows from index $start_idx to $end_idx."            help = "Learning rate for optimization stage 2."    final_processed = processed[]

    results = Vector{Dict{Symbol,Any}}(undef, num_windows_to_process)

                arg_type = Float64    final_errors = errors[]

    # Use a sequential loop for debugging

    for i in 1:num_windows_to_process            default = 2.5e-3

        window_data = windows_to_process[i]

        global_id = start_idx + i - 1        "--output"    @info "Processamento paralelo concluído" processed=final_processed errors=final_errors

        signal_start = 1 + (global_id - 1) * cfg.step

                    help = "Path for the output CSV file."

        results[i] = run_window_optimized(global_id, signal_start, window_data, cfg)

    end            arg_type = String    if final_errors > 0



    df = DataFrame(filter(d -> !haskey(d, :error) || d[:error] == "", results))            default = "results/mawi_metrics.csv"        @warn "$final_errors janelas falharam - revise os resultados"

    if !empty(df)

        ensure_output_dir(cfg.output_path)        "--warm-start-haar"    end

        CSV.write(cfg.output_path, df)

        @info "Successfully processed $(nrow(df)) windows. Results saved to $(cfg.output_path)."            help = "Initialize MERA with Haar wavelet filters."

    else

        @warn "No windows were processed successfully."            action = :store_true    # Mostrar estatísticas do cache

    end

end        "--use-gpu"    cache_stats = Wave6G.get_cache_stats()



main()            help = "Enable GPU acceleration."    @info "Estatísticas do Cache Inteligente" hits=cache_stats.hits misses=cache_stats.misses hit_rate=cache_stats.hit_rate


            action = :store_true

    end    df = DataFrame(rows)

    parsed = parse_args(s)    ensure_output_dir(cfg.output_path)

        CSV.write(cfg.output_path, df)

    if parsed["use-gpu"]    @info "Resultados salvos" output=cfg.output_path rows=nrow(df) success_rate="$(round((final_processed - final_errors) / final_processed * 100, digits=1))%"

        setup_gpu()end

    else

        USE_GPU[] = falseif abspath(PROGRAM_FILE) == @__FILE__

        @info "GPU usage is disabled by command-line argument."    main()

    endend


    init_mode = parsed["warm-start-haar"] ? :haar : :random
    return ExperimentConfig(
        parsed["data"],
        parsed["window-size"],
        parsed["step"],
        parsed["retain"],
        max(parsed["start-window"], 1),
        parsed["num-windows"],
        parsed["mera-L"],
        parsed["mera-chi"],
        parsed["mera-chimid"],
        parsed["stage1-iters"],
        parsed["stage1-lr"],
        parsed["stage2-iters"],
        parsed["stage2-lr"],
        parsed["output"],
        init_mode
    )
end

psnr(original, reconstructed) = begin
    mse = mean(abs2, reconstructed .- original)
    max_val = maximum(abs.(original))
    mse < eps() || max_val == 0 ? Inf : 10.0 * log10(max_val^2 / mse)
end

to_device(data) = USE_GPU[] ? cu(data) : data
from_device(data) = data isa CUDA.AbstractGPUArray ? Array(data) : data

function run_window_optimized(window_id::Int, start_idx::Int, signal::AbstractVector{<:Number}, cfg::ExperimentConfig)
    try
        println("Starting window $window_id on thread $(Threads.threadid())")
        original = to_device(Float64.(signal))
        
        schedule = [
            (L=cfg.mera_L, chi=cfg.mera_chi, chimid=cfg.mera_chimid, numiter=cfg.stage1_iters, lr=cfg.stage1_lr, reinit=true, init=cfg.init_mode),
            (numiter=cfg.stage2_iters, lr=cfg.stage2_lr, reinit=false)
        ]

        mera_time = @elapsed begin
            println("Preparing variational state for window $window_id...")
            state = Wave6G.prepare_variational_state(original; L=cfg.mera_L, chi=cfg.mera_chi, chimid=cfg.mera_chimid, normalize=true, init=cfg.init_mode)
            state = to_device(state)
            println("Optimizing MERA for window $window_id...")
            state, loss_history = Wave6G.optimize_variational_schedule!(state, schedule)
            println("Extracting coefficients for window $window_id...")
            coeffs_mera = Wave6G.mera_coeffs(state)
            println("Thresholding coefficients for window $window_id...")
            filtered_mera, kept_mera, total_mera = Wave6G.threshold_coeffs(coeffs_mera, cfg.retain_ratio)
            println("Reconstructing signal for window $window_id...")
            reconstructed_mera = Wave6G.mera_reconstruct(state, filtered_mera)
        end
        
        psnr_mera = psnr(from_device(original), from_device(reconstructed_mera))
        println("Finished window $window_id. PSNR: $psnr_mera, Time: $mera_time")
        
        return Dict(
            :window_id => window_id,
            :psnr_mera => psnr_mera,
            :time_mera => mera_time,
            :kept_mera => kept_mera,
            :total_mera => total_mera,
            :error => ""
        )
    catch e
        @error "Error processing window" window_id start_idx exception=(e, catch_backtrace())
        return Dict(:window_id => window_id, :error => string(e))
    end
end

function ensure_output_dir(path::String)
    dir = dirname(path)
    isempty(dir) || dir == "." || isdir(dir) || mkpath(dir)
end

function main()
    cfg = parse_cli()
    @info "Configuration loaded. Preparing dataset..." cfg
    dataset = MAWI.prepare_window_dataset(cfg.data_path; window_size=cfg.window_size, step=cfg.step, normalize=true)
    
    all_windows = dataset.windows
    total_available = length(all_windows)
    start_idx = min(cfg.start_window, total_available)
    
    end_idx = cfg.num_windows > 0 ? min(start_idx + cfg.num_windows - 1, total_available) : total_available
    windows_to_process = all_windows[start_idx:end_idx]
    num_windows_to_process = length(windows_to_process)

    if num_windows_to_process == 0
        @info "No windows to process. Exiting."
        return
    end

    @info "Processing $num_windows_to_process windows from index $start_idx to $end_idx."
    results = Vector{Dict{Symbol,Any}}(undef, num_windows_to_process)
    
    Threads.@threads for i in 1:num_windows_to_process
        window_data = windows_to_process[i]
        global_id = start_idx + i - 1
        signal_start = 1 + (global_id - 1) * cfg.step
        
        results[i] = run_window_optimized(global_id, signal_start, window_data, cfg)
    end

    df = DataFrame(filter(d -> !haskey(d, :error) || d[:error] == "", results))
    if !empty(df)
        ensure_output_dir(cfg.output_path)
        CSV.write(cfg.output_path, df)
        @info "Successfully processed $(nrow(df)) windows. Results saved to $(cfg.output_path)."
    else
        @warn "No windows were processed successfully."
    end
end

main()
