# Four kernels, launched twice each inside one profiling window, whose `hw_counter_limiter` label the
# suite checks. Compute- and memory-bound are known by construction. The other two run the same loads
# and differ in how many blocks are resident (and in hops), so telling them apart checks the `occupancy_low`
# cutoff, not a limiter; the size of `cal_latency` was chosen after its first label. NVIDIA only.
using KernelAbstractions, Adapt, CUDA, GPUDiagnostics
backend = CUDABackend()

# Arithmetic only: a long FMA chain per work-item, one store at the end.
@kernel function cal_compute!(out, niter)
    i = @index(Global, Linear)
    a = Float32(i) * 1.0f-6
    for _ in Int32(1):niter
        a = fma(a, 0.999f0, 0.001f0)
    end
    @inbounds out[i] = a
end
# Streaming through arrays far larger than L2.
@kernel function cal_memory!(out, @Const(x), @Const(y))
    i = @index(Global, Linear)
    @inbounds out[i] = x[i] + 2.0f0 * y[i]
end
# Dependent loads through a random cycle; the 32 lanes of a warp load the same element, so every
# warp has one small request outstanding and waits for it. Launched at HALF the GPU's resident
# blocks: at full occupancy the requests queue and the memory throughput is close to `low`
# (41 % on an A100 at 1410 MHz, where the label is `:mixed`).
@kernel function cal_latency!(out, @Const(next), nhops)
    i = @index(Global, Linear)
    p = Int32(mod((i - 1) ÷ 32 * 7919, length(next)) + 1)
    for _ in Int32(1):nhops
        @inbounds p = next[p]
    end
    @inbounds out[i] = p
end
# The same dependent loads in a launch of one 256-item block: 8 warps on one SM.
@kernel function cal_occupancy!(out, @Const(next), nhops)
    i = @index(Global, Linear)
    p = Int32(mod((i - 1) ÷ 32 * 7919, length(next)) + 1)
    for _ in Int32(1):nhops
        @inbounds p = next[p]
    end
    @inbounds out[i] = p
end

function random_cycle(n)   # Sattolo: one cycle through all n elements
    p = collect(Int32(1):Int32(n))
    for i in n:-1:2
        j = rand(1:(i - 1)); p[i], p[j] = p[j], p[i]
    end
    return p
end

nthreads = 2^20
nbig = 2^27                                   # 512 MiB per array
x = KernelAbstractions.ones(backend, Float32, nbig); y = copy(x); big = similar(x)
next = Adapt.adapt(backend, random_cycle(2^26))   # 256 MiB
fout = KernelAbstractions.zeros(backend, Float32, nthreads)
iout = KernelAbstractions.zeros(backend, Int32, nthreads)
half_blocks = max(1, gpu_sm_count(backend) * gpu_max_threads_per_sm(backend) ÷ 512)
function all_kernels()
    cal_compute!(backend, 256)(fout, Int32(4096); ndrange = nthreads)
    cal_memory!(backend, 256)(big, x, y; ndrange = nbig)
    cal_latency!(backend, 256)(iout, next, Int32(2000); ndrange = 256 * half_blocks)
    cal_occupancy!(backend, 256)(iout, next, Int32(1000); ndrange = 256)
    KernelAbstractions.synchronize(backend)
end
all_kernels()   # compile outside the window
hw_counter_window(backend) do
    all_kernels(); all_kernels()
end
println("profiler_attached=", something(profiler_attached(), "nothing"))
