# Attainable vector floating-point peak of a backend, MEASURED — the denominator of any
# percent-of-peak figure for scalar kernels. No per-architecture table: every thread runs several
# independent chains of dependent FMAs (the instruction the production kernels are made of), the
# launch is sized to fill the device, and the best of a few trials is the ceiling scalar code can
# attain on that device at the clocks it actually holds. A matrix-multiply probe would be routed
# to the tensor/matrix units on datacenter parts (2× the vector FP64 rate on A100/H100/MI300X) —
# the wrong yardstick for scalar kernels — which is why the probe is an FMA chain. The GEMM rate
# is still worth having as an explicit UPPER reference (`measure_gemm_flops`), never as the peak.
#
# Latency hiding is a geometry question, and the right geometry differs per architecture: the
# chains per thread × the resident threads per SM must exceed lanes × FMA latency, and a launch
# that is not a whole number of resident waves leaves a partial last round. With one fixed
# geometry (8 chains, 2^20 threads) an A100 read 54 % of its spec while an H100 read 91 % at the
# same clock, so the probe now SWEEPS a small grid — chains ∈ {4, 8, 16} × threads ∈ {1, 2, 4} ×
# the device's resident-thread capacity — with one trial per point, then re-measures the best
# point with the full trial count. `peak_flops_probe` returns the whole record (the winning
# geometry, the sweep, the launch time) so a manifest can say HOW its peak was measured;
# `measure_peak_flops` is the number alone.
#
# The multiplier/addend derive from a per-thread seed so nothing folds at compile time, and the
# recurrence converges (|x| < 1) so no overflow or denormal slow paths distort the timing. The
# element type is generic: the same kernel measures FP64 and FP32 (and FP16 where a backend
# computes in it).
#
# The CPU backend runs the same probe. It under-reports the host by the SIMD width (one scalar
# chain per work-item); `LinearAlgebra.peakflops` is the vectorised host number — a different
# quantity, documented as the alternative rather than wrapped here.

const _PEAK_CHAINS = (4, 8, 16)          # default chain-count grid
const _PEAK_THREAD_FILLS = (1, 2, 4)     # default launch sizes, in units of the resident capacity
const _PEAK_THREADS_FALLBACK = 2^20      # launch size when the backend cannot report its capacity

# Pairwise tree sum of an NTuple — the same association in the kernel and in the host reference,
# so the oracle compares bit-for-bit (up to the FMA rounding both sides share).
@inline _pairsum(a::NTuple{1}) = a[1]
@inline _pairsum(a::NTuple{2}) = a[1] + a[2]
@inline function _pairsum(a::NTuple{N}) where {N}
    h = N ÷ 2
    return _pairsum(ntuple(k -> a[k], Val(h))) + _pairsum(ntuple(k -> a[h + k], Val(N - h)))
end

@kernel function _fma_chain_kernel!(out, @Const(seed), n_iters, ::Val{C}) where {C}
    i = @index(Global, Linear)
    @inbounds begin
        s = seed[i]
        T = typeof(s)
        x = T(0.999_999) - T(1.0e-9) * s      # |x| < 1: the recurrence converges to y / (1 − x)
        y = T(1.0e-3) + T(1.0e-9) * s
        a = ntuple(k -> s + T(k - 1), Val(C))
        for _ in 1:n_iters
            a = map(v -> fma(v, x, y), a)     # C independent chains, unrolled (C is a type parameter)
        end
        out[i] = _pairsum(a)
    end
end

# Host reference of one thread's result (the test oracle: proves the kernel ran the chains it is
# credited with). Same operations in the same order as the kernel.
function _fma_chain_reference(s::T, n_iters::Integer; chains::Integer = 8) where {T <: AbstractFloat}
    x = T(0.999_999) - T(1.0e-9) * s
    y = T(1.0e-3) + T(1.0e-9) * s
    a = ntuple(k -> s + T(k - 1), Int(chains))
    for _ in 1:n_iters
        a = map(v -> fma(v, x, y), a)
    end
    return _pairsum(a)
end

function _fma_chain_run!(backend, out, seed, n_iters, workgroup, ::Val{C}) where {C}
    kern = _fma_chain_kernel!(backend, workgroup)
    KernelAbstractions.synchronize(backend)
    t0 = time_ns()
    kern(out, seed, Int32(n_iters), Val(C); ndrange = length(out))
    KernelAbstractions.synchronize(backend)
    return (time_ns() - t0) * 1.0e-9
end

"""
    PeakProbe

Result of [`peak_flops_probe`](@ref): `flops` (the attainable rate, FLOP/s) and the geometry that
attained it — `chains` per thread, `n_threads`, `workgroup`, `n_iters` per chain, `best_s` (the
fastest of `trials` launches at that geometry) — plus `capacity` (the device's resident-thread
capacity the launch sizes were derived from, `missing` when the backend cannot report it) and
the sweep that chose the geometry as three parallel vectors `sweep_chains`, `sweep_n_threads`,
`sweep_flops` (one trial per point). `Float64(p)` is `p.flops`.
"""
struct PeakProbe
    eltype::Symbol
    flops::Float64
    chains::Int
    n_threads::Int
    workgroup::Int
    n_iters::Int
    best_s::Float64
    trials::Int
    capacity::Union{Missing, Int}
    sweep_chains::Vector{Int}
    sweep_n_threads::Vector{Int}
    sweep_flops::Vector{Float64}
end
Base.Float64(p::PeakProbe) = p.flops

# Resident-thread capacity of the current device, or `missing` when the backend has no
# `:device_props` (the CPU backend, a vendor-less backend).
function _peak_capacity(backend::Backend)
    try
        return Int(gpu_sm_count(backend)) * Int(gpu_max_threads_per_sm(backend))
    catch err
        err isa BackendUnsupported && return missing
        rethrow()
    end
end

_peak_grid(x::Integer) = (Int(x),)
_peak_grid(x::Union{Tuple, AbstractVector}) = Tuple(Int.(x))

# One geometry, timed: warm-up + calibration launch, then `trials` launches at the calibrated
# length; returns (flops, n_iters, best_s). The oracle check is done here so every sweep point
# is verified, not only the winner.
function _peak_point(backend, ::Type{T}, chains::Int, n_threads::Int, workgroup::Int, trials::Int,
        target_seconds::Real, buffers) where {T}
    seed, out = get!(buffers, n_threads) do
        (Adapt.adapt(backend, T.(0:(n_threads - 1))), Adapt.adapt(backend, zeros(T, n_threads)))
    end
    C = Val(chains)
    n0 = 256
    _fma_chain_run!(backend, out, seed, n0, workgroup, C)                  # warm-up (compile)
    t0 = _fma_chain_run!(backend, out, seed, n0, workgroup, C)
    n_iters = clamp(round(Int, n0 * target_seconds / max(t0, 1.0e-6)), n0, typemax(Int32) ÷ 2)
    best = Inf
    for _ in 1:trials
        best = min(best, _fma_chain_run!(backend, out, seed, n_iters, workgroup, C))
    end
    # Sanity: the device computed the chains it is credited with (guards a mis-launched kernel
    # or a compiler that hoisted the loop).
    h = Array(out)
    ref = _fma_chain_reference(zero(T), n_iters; chains)
    isapprox(h[1], ref; rtol = sqrt(eps(T))) ||
        error("peak_flops_probe: kernel result $(h[1]) ≠ host reference $ref at $chains chains — the probe did not run as credited")
    return (2.0 * chains * n_iters * n_threads / best, n_iters, best)
end

"""
    peak_flops_probe(backend, T = Float64; chains = $(_PEAK_CHAINS), n_threads = :auto, workgroup = 256,
                     trials = 5, target_seconds = 0.2) -> PeakProbe

Attainable vector peak of `backend` for element type `T`, measured with a dependent-FMA-chain
kernel over a small geometry sweep: every combination of `chains` (independent accumulators per
thread) and `n_threads` is launched once with the chain length calibrated to about
`target_seconds`; the fastest geometry is then re-measured as the best of `trials` launches.
`n_threads = :auto` derives the launch sizes from the device's resident-thread capacity
(`gpu_sm_count × gpu_max_threads_per_sm`, times $(_PEAK_THREAD_FILLS)) so every launch is a whole
number of resident waves; a backend that cannot report its capacity gets $(_PEAK_THREADS_FALLBACK)
threads. An `Integer` or a tuple for either keyword pins that axis. FLOP = threads × chains ×
iterations × 2 (one FMA = 2 FLOP). Never routed to matrix/tensor units: this is the ceiling for
scalar code in `T` at the clocks the device holds during the probe (a power-managed part reports
its power-limited peak; sample the clock alongside with `with_gpu_sampler` to label it). The
GEMM ceiling for the same device is [`measure_gemm_flops`](@ref). Costs roughly
`(points + trials) × target_seconds` of device time; the default sweep is nine points.

Requires the `:peak_flops` capability, and `:fp64` for `T = Float64`. On the CPU backend the
probe measures one scalar chain per work-item, which under-reports the host by its SIMD width;
`LinearAlgebra.peakflops` is the vectorised host number.
"""
function peak_flops_probe(backend::Backend, ::Type{T} = Float64; chains = _PEAK_CHAINS,
        n_threads = :auto, workgroup::Integer = 256, trials::Integer = 5,
        target_seconds::Real = 0.2) where {T <: AbstractFloat}
    _require(backend, :peak_flops, :peak_flops_probe)
    T === Float64 && _require(backend, :fp64, :peak_flops_probe)
    workgroup > 0 && trials > 0 && target_seconds > 0 ||
        throw(ArgumentError("peak_flops_probe: workgroup, trials, target_seconds must be > 0"))
    cgrid = _peak_grid(chains)
    all(c -> 1 ≤ c ≤ 64, cgrid) || throw(ArgumentError("peak_flops_probe: chains must be in 1:64, got $cgrid"))
    capacity = _peak_capacity(backend)
    tgrid = n_threads === :auto ?
        (capacity === missing ? (_PEAK_THREADS_FALLBACK,) : Tuple(f * capacity for f in _PEAK_THREAD_FILLS)) :
        _peak_grid(n_threads)
    all(>(0), tgrid) || throw(ArgumentError("peak_flops_probe: n_threads must be > 0, got $tgrid"))
    buffers = Dict{Int, Any}()
    sc = Int[]; st = Int[]; sf = Float64[]
    for c in cgrid, n in tgrid
        f, _, _ = _peak_point(backend, T, c, n, Int(workgroup), 1, target_seconds, buffers)
        push!(sc, c); push!(st, n); push!(sf, f)
    end
    k = argmax(sf)
    f, n_iters, best = _peak_point(backend, T, sc[k], st[k], Int(workgroup), Int(trials), target_seconds, buffers)
    f = max(f, sf[k])   # best-of over everything this geometry ran
    return PeakProbe(nameof(T), f, sc[k], st[k], Int(workgroup), n_iters, best, Int(trials), capacity, sc, st, sf)
end

"""
    measure_peak_flops(backend, T = Float64; kwargs...) -> Float64

The attainable vector peak in FLOP/s: [`peak_flops_probe`](@ref)`(backend, T; kwargs...).flops`.
"""
measure_peak_flops(backend::Backend, ::Type{T} = Float64; kwargs...) where {T <: AbstractFloat} =
    peak_flops_probe(backend, T; kwargs...).flops

"""
    measure_gemm_flops(backend, T = Float64; n = 4096, trials = 3) -> Float64

Matrix-multiply rate of `backend` in FLOP/s: `mul!(C, A, B)` on `n × n` matrices of `T` through
the array type's BLAS (cuBLAS, rocBLAS, the host BLAS on the CPU backend), best of `trials` after
one warm-up, counted as `2n³` FLOP. On datacenter parts this is routed to the matrix/tensor units
and exceeds the vector rate (about 2× for FP64 on A100 / H100 / MI300X); it is the UPPER reference
for what the silicon can do in `T`, not the ceiling for scalar kernels — that is
[`peak_flops_probe`](@ref). Requires `:peak_flops`, and `:fp64` for `T = Float64`.
"""
function measure_gemm_flops(backend::Backend, ::Type{T} = Float64; n::Integer = 4096,
        trials::Integer = 3) where {T <: AbstractFloat}
    _require(backend, :peak_flops, :measure_gemm_flops)
    T === Float64 && _require(backend, :fp64, :measure_gemm_flops)
    n > 0 && trials > 0 || throw(ArgumentError("measure_gemm_flops: n and trials must be > 0"))
    A = Adapt.adapt(backend, rand(T, n, n)); B = Adapt.adapt(backend, rand(T, n, n))
    Cm = Adapt.adapt(backend, zeros(T, n, n))
    best = Inf
    for k in 0:trials     # k = 0 is the warm-up
        KernelAbstractions.synchronize(backend)
        t0 = time_ns()
        LinearAlgebra.mul!(Cm, A, B)
        KernelAbstractions.synchronize(backend)
        k == 0 || (best = min(best, (time_ns() - t0) * 1.0e-9))
    end
    return 2.0 * n^3 / best
end
