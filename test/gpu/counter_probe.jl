# A bounded collection workload; filter the profiler to counter_probe and skip
# the first matching launch. Run with GPUDIAGNOSTICS_GPU=cuda or rocm.
using KernelAbstractions, Adapt, Test
vendor = get(ENV, "GPUDIAGNOSTICS_GPU", "cuda")
if vendor == "cuda"
    using CUDA
    backend = CUDABackend()
elseif vendor == "rocm"
    using AMDGPU
    backend = ROCBackend()
else
    error("GPUDIAGNOSTICS_GPU must be cuda or rocm")
end

@kernel function counter_probe!(out, @Const(x), niter)
    i = @index(Global, Linear)
    @inbounds begin
        a = x[i]
        for _ in 1:niter
            a = fma(a, 0.999, 0.001)
        end
        out[i] = a
    end
end

n, niter = 4096, 32
x = Adapt.adapt(backend, fill(0.5, n))
out = similar(x)
for _ in 1:3
    counter_probe!(backend, 256)(out, x, Int32(niter); ndrange = n)
    KernelAbstractions.synchronize(backend)
end
reference = foldl((a, _) -> fma(a, 0.999, 0.001), 1:niter; init = 0.5)
@test all(v -> isapprox(v, reference; rtol = 1e-12), Array(out))
