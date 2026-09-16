# peak_report.jl — the measured FP64 / FP32 peak of every device the vendor package sees, with
# the geometry that attained it, the GEMM reference and the clock/power the sampler saw during
# the probe. The table issue #28 asks for; run it on any card and paste the output there.
#
#     GPUDIAGNOSTICS_GPU=cuda julia --project=test/gpu test/gpu/peak_report.jl
#     GPUDIAGNOSTICS_GPU=rocm …
#
# Prints, per device: name, resident capacity, the PeakProbe (winner + sweep), the FP32 probe,
# the GEMM rate, and the busy-window median SM clock and mean power from a sampler child
# (skipped, with a note, where the backend has no telemetry). Nothing is written to disk.
using KernelAbstractions, Adapt
using GPUDiagnostics

const VENDOR = lowercase(get(ENV, "GPUDIAGNOSTICS_GPU", ""))
VENDOR in ("cuda", "rocm") || error("set GPUDIAGNOSTICS_GPU=cuda or rocm")
if VENDOR == "cuda"
    using CUDA
    backend = CUDABackend()
else
    using AMDGPU
    backend = ROCBackend()
end

tflops(x) = string(round(x / 1e12; digits = 3), " TFLOP/s")

for dev in 1:gpu_device_count(backend)
    gpu_device!(backend, dev)
    println("\n== device $dev: ", gpu_name(backend), " (", gpu_arch(backend), "), ",
        gpu_sm_count(backend), " SMs × ", gpu_max_threads_per_sm(backend), " threads")
    pr = nothing
    telem = nothing
    try
        pr, telem = with_gpu_sampler(backend, 0.1; devices = [dev]) do
            peak_flops_probe(backend; trials = 5)
        end
    catch err
        err isa BackendUnsupported || rethrow()
        println("  (no sampler: ", sprint(showerror, err), ")")
        pr = peak_flops_probe(backend; trials = 5)
    end
    show(stdout, MIME"text/plain"(), pr); println()
    if telem !== nothing && telem.ticks > 0
        st = gpu_telemetry_stats(telem)
        clk = get(st, "sm_clock_MHz_busy_median", missing)
        pw = get(st, "power_W_busy_mean", missing)
        cap = get(st, "power_limit_W_mean", missing)
        r1(x) = x === missing ? "missing" : string(round(x; digits = 1))
        println("  during the probe: SM clock busy median ", r1(clk), " MHz, power ", r1(pw), " W (cap ", r1(cap), " W), ",
            st["busy_samples"], " busy samples")
    end
    p32 = peak_flops_probe(backend, Float32; trials = 3)
    println("  FP32: ", tflops(p32.flops), " at ", p32.chains, " chains × ", p32.n_threads, " threads")
    g = measure_gemm_flops(backend; n = 4096, trials = 3)
    println("  GEMM FP64 (n = 4096): ", tflops(g), " = ", round(g / pr.flops; digits = 2), "× the vector probe")
end
