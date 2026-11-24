# A bounded multi-kernel workload for the profiling window: warm-up launches outside the window,
# then 2 launches of `window_fma!` and 1 of `window_copy!` inside `hw_counter_window`, and a peak
# probe that must know it runs under the collector. Run with GPUDIAGNOSTICS_GPU=cuda.
using KernelAbstractions, Adapt, CUDA, GPUDiagnostics
backend = CUDABackend()

@kernel function window_fma!(out, @Const(x), niter)
    i = @index(Global, Linear)
    @inbounds begin
        a = x[i]
        for _ in 1:niter
            a = fma(a, 0.999f0, 0.001f0)
        end
        out[i] = a
    end
end
@kernel function window_copy!(out, @Const(x))
    i = @index(Global, Linear)
    @inbounds out[i] = x[i]
end

n = 2^20
x = Adapt.adapt(backend, fill(0.5f0, n))
out = similar(x)
step!() = (window_fma!(backend, 256)(out, x, Int32(64); ndrange = n); window_copy!(backend, 256)(out, x; ndrange = n))
for _ in 1:5   # warm-up: never profiled under profile_from_start = false
    step!()
end
hw_counter_window(backend) do
    step!()
    window_fma!(backend, 256)(out, x, Int32(64); ndrange = n)
end
step!()        # after the window: not profiled either
KernelAbstractions.synchronize(backend)
println("probe_under_profiler=", peak_flops_probe(backend, Float32).under_profiler)
println("profiler_attached=", something(profiler_attached(), "nothing"))
withenv(GPUDiagnostics.PROFILER_ENV => nothing) do   # what a hand-started ncu leaves: the Nsight variables
    println("profiler_attached_without_marker=", something(profiler_attached(), "nothing"))
end
