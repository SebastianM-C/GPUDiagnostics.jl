# Attainable vector floating-point peak of a backend, MEASURED — the denominator of any
# percent-of-peak figure for scalar kernels. No per-architecture table: every thread runs several
# independent chains of dependent FMAs (the instruction the production kernels are made of), the
# launch is sized to fill the device, and the best of a few trials is the ceiling scalar code can
# attain on that device at the clocks it actually holds. A matrix-multiply probe would be routed
# to the tensor/matrix units on datacenter parts (2× the vector FP64 rate on A100/H100/MI300X) —
# the wrong yardstick for scalar kernels — which is why the probe is an FMA chain.
#
# Latency hiding: `CHAINS` independent accumulators per thread × the resident threads per SM
# (≥ 1024 at the default launch) ≫ lanes × FMA latency (64 × ~8 for FP64 on Hopper), so the
# vector pipes stay full without relying on cross-thread scheduling. The multiplier/addend derive
# from a per-thread seed so nothing folds at compile time, and the recurrence converges (|x| < 1)
# so no overflow or denormal slow paths distort the timing. The element type is generic: the same
# kernel measures FP64 and FP32 (and FP16 where a backend computes in it).
#
# The CPU backend runs the same probe. It under-reports the host by the SIMD width (one scalar
# chain per work-item); `LinearAlgebra.peakflops` is the vectorised host number — a different
# quantity, documented as the alternative rather than wrapped here.

const _PEAK_CHAINS = 8

@kernel function _fma_chain_kernel!(out, @Const(seed), n_iters)
    i = @index(Global, Linear)
    @inbounds begin
        s = seed[i]
        T = typeof(s)
        x = T(0.999_999) - T(1.0e-9) * s      # |x| < 1: the recurrence converges to y / (1 − x)
        y = T(1.0e-3) + T(1.0e-9) * s
        a1 = s; a2 = s + T(1); a3 = s + T(2); a4 = s + T(3)
        a5 = s + T(4); a6 = s + T(5); a7 = s + T(6); a8 = s + T(7)
        for _ in 1:n_iters
            a1 = fma(a1, x, y); a2 = fma(a2, x, y); a3 = fma(a3, x, y); a4 = fma(a4, x, y)
            a5 = fma(a5, x, y); a6 = fma(a6, x, y); a7 = fma(a7, x, y); a8 = fma(a8, x, y)
        end
        out[i] = ((a1 + a2) + (a3 + a4)) + ((a5 + a6) + (a7 + a8))
    end
end

# Host reference of one thread's result (the test oracle: proves the kernel ran the chains it is
# credited with). Same operations in the same order as the kernel.
function _fma_chain_reference(s::T, n_iters::Integer) where {T <: AbstractFloat}
    x = T(0.999_999) - T(1.0e-9) * s
    y = T(1.0e-3) + T(1.0e-9) * s
    a = ntuple(k -> s + T(k - 1), _PEAK_CHAINS)
    for _ in 1:n_iters
        a = map(v -> fma(v, x, y), a)
    end
    return ((a[1] + a[2]) + (a[3] + a[4])) + ((a[5] + a[6]) + (a[7] + a[8]))
end

function _fma_chain_run!(backend, out, seed, n_iters, workgroup)
    kern = _fma_chain_kernel!(backend, workgroup)
    KernelAbstractions.synchronize(backend)
    t0 = time_ns()
    kern(out, seed, Int32(n_iters); ndrange = length(out))
    KernelAbstractions.synchronize(backend)
    return (time_ns() - t0) * 1.0e-9
end

"""
    measure_peak_flops(backend, T = Float64; n_threads = 2^20, workgroup = 256, trials = 5,
                       target_seconds = 0.2) -> Float64

Attainable vector peak of `backend` for element type `T` in FLOP/s, measured with a
dependent-FMA-chain kernel (`$(_PEAK_CHAINS)` independent chains per thread over `n_threads`
threads). The chain length is calibrated so one launch lasts about `target_seconds`, then the
best of `trials` launches is returned (best-of, like `LinearAlgebra.peakflops`, discards
clock-ramp and scheduling noise). FLOP = threads × chains × iterations × 2 (one FMA = 2 FLOP).
Never routed to matrix/tensor units: this is the ceiling for scalar code in `T`. Costs about
`(trials + 2) × target_seconds` of device time.

Requires the `:peak_flops` capability, and `:fp64` for `T = Float64`. On the CPU backend the
probe measures one scalar chain per work-item, which under-reports the host by its SIMD width;
`LinearAlgebra.peakflops` is the vectorised host number.
"""
function measure_peak_flops(backend::Backend, ::Type{T} = Float64; n_threads::Integer = 2^20,
        workgroup::Integer = 256, trials::Integer = 5, target_seconds::Real = 0.2) where {T <: AbstractFloat}
    _require(backend, :peak_flops, :measure_peak_flops)
    T === Float64 && _require(backend, :fp64, :measure_peak_flops)
    n_threads > 0 && workgroup > 0 && trials > 0 && target_seconds > 0 ||
        throw(ArgumentError("measure_peak_flops: n_threads, workgroup, trials, target_seconds must be > 0"))
    seed = Adapt.adapt(backend, T.(0:(n_threads - 1)))
    out = Adapt.adapt(backend, zeros(T, n_threads))
    flop(n_iters) = 2.0 * _PEAK_CHAINS * n_iters * n_threads
    # Calibrate: a short launch (also the JIT warm-up), then size n_iters to ≈ target_seconds.
    n0 = 256
    _fma_chain_run!(backend, out, seed, n0, workgroup)                     # warm-up (compile)
    t0 = _fma_chain_run!(backend, out, seed, n0, workgroup)
    n_iters = clamp(round(Int, n0 * target_seconds / max(t0, 1.0e-6)), n0, typemax(Int32) ÷ 2)
    best = Inf
    for _ in 1:trials
        best = min(best, _fma_chain_run!(backend, out, seed, n_iters, workgroup))
    end
    # Sanity: the device computed the chains it is credited with (guards a mis-launched kernel
    # or a compiler that hoisted the loop).
    h = Array(out)
    ref = _fma_chain_reference(zero(T), n_iters)
    isapprox(h[1], ref; rtol = sqrt(eps(T))) ||
        error("measure_peak_flops: kernel result $(h[1]) ≠ host reference $ref — the probe did not run as credited")
    return flop(n_iters) / best
end
