# Real-hardware tests: the conformance suite plus the vendor paths the CPU tests cannot reach.
# Not part of `Pkg.test` (no GPU in CI); run by hand on a box with a GPU:
#
#     GPUDIAGNOSTICS_GPU=cuda julia --project=test/gpu -e 'using Pkg; Pkg.resolve(); Pkg.instantiate(); include("test/gpu/runtests.jl")'
#     GPUDIAGNOSTICS_GPU=rocm …
#
# `Pkg.resolve()` first: the env dev-tracks the package, and a manifest resolved before a dependency
# was added to Project.toml is not refreshed by `instantiate` alone.
# One vendor per process (loading both vendor packages is not the point). The kernel under
# test is a small KA kernel with a hot loop, so the resource report, the native / IR mix and
# the FP64 issue floor all have something to say.

using Test, KernelAbstractions, Adapt
using GPUDiagnostics
include(joinpath(@__DIR__, "..", "conformance.jl"))

const VENDOR = lowercase(get(ENV, "GPUDIAGNOSTICS_GPU", ""))
VENDOR in ("cuda", "rocm") || error("set GPUDIAGNOSTICS_GPU=cuda or rocm")
if VENDOR == "cuda"
    using CUDA
    const backend = CUDABackend()
    const isa_vendor = :nvidia
    const cross_target = "sm_90"
else
    using AMDGPU
    const backend = ROCBackend()
    const isa_vendor = :amd
    const cross_target = "gfx942"
end

@kernel function gpudiag_probe_kernel!(out, @Const(x), n_iters)
    i = @index(Global, Linear)
    @inbounds begin
        a = x[i]; b = 0.5 * a
        for _ in 1:n_iters
            a = fma(a, 0.999, 1.0e-3); b = fma(b, a, 1.0e-4)
        end
        out[i] = a + sqrt(abs(b))
    end
end

@testset "GPUDiagnostics on $(gpu_name(backend)) ($(gpu_arch(backend)))" begin
    n = 2^16
    x = Adapt.adapt(backend, rand(Float64, n))
    out = Adapt.adapt(backend, zeros(Float64, n))
    gpudiag_probe_kernel!(backend, 256)(out, x, Int32(100); ndrange = n)
    KernelAbstractions.synchronize(backend)
    cks = compiled_kernels(backend; pattern = r"gpudiag_probe_kernel")
    @test length(cks) == 1
    ck = only(cks)

    conformance(backend; kernel = ck)

    @testset "device API" begin
        @test capabilities(backend) ⊇ [:devices, :device_props, :events, :telemetry, :peak_flops, :fp64,
            :kernel_inventory, :resources, :occupancy, :native_mix, :ir_mix]
        @test gpu_device_count(backend) ≥ 1 && gpu_sm_count(backend) > 0 && gpu_max_threads_per_sm(backend) > 0
        @test 0 < thread_fill_occupancy(backend, n) < 1000
        m = gpu_memory_info(backend)
        @test m.total > m.free > 0 && m.used ≥ 0
        @test gpu_power(backend) > 0
        u = gpu_utilization(backend)
        @test 0 ≤ u.compute ≤ 1 && 0 ≤ u.memory ≤ 1
    end

    @testset "events + LaunchTimer" begin
        timer = LaunchTimer()
        lane = launch_lane(timer, backend)
        for _ in 1:5
            e0 = launch_tick(timer, backend)
            gpudiag_probe_kernel!(backend, 256)(out, x, Int32(1000); ndrange = n)
            launch_tock!(timer, lane, backend, e0)
        end
        lt = launch_times(timer)
        @test length(only(values(lt))) == 5 && all(>(0), only(values(lt)))
    end

    @testset "measured peak" begin
        p64 = measure_peak_flops(backend; trials = 2, target_seconds = 0.05)
        p32 = measure_peak_flops(backend, Float32; trials = 2, target_seconds = 0.05)
        @test 1e9 < p64 < 1e15 && 1e9 < p32 < 1e16
        @info "measured peaks" fp64_TFLOPs = round(p64 / 1e12; digits = 2) fp32_TFLOPs = round(p32 / 1e12; digits = 2)
    end

    @testset "resource report" begin
        r = kernel_resources(backend, ck)
        @test r isa KernelResources && r.block_size == 256
        @test !ismissing(r.registers) && r.registers > 0
        @test !ismissing(r.local_mem_bytes) && !ismissing(r.shared_mem_bytes)
        @test r.occupancy isa KernelOccupancy && 0 < r.occupancy.fraction ≤ 1
        VENDOR == "rocm" && @test ismissing(r.const_mem_bytes)     # HIP has no CONST_SIZE_BYTES attribute
        VENDOR == "cuda" && @test !ismissing(r.const_mem_bytes)
        @test !isempty(r.isa)
        txt = sprint(show, MIME"text/plain"(), r)
        @test occursin("registers", txt) && occursin("warps per SM", txt)
        @info "kernel_resources" report = txt
    end

    @testset "instruction mix: native, IR, cross-compiled, FP64 floor" begin
        m = kernel_instruction_mix(backend, ck)
        @test m.vendor === isa_vendor && m.native && m.coverage ≥ 0.98 && m.total > 0
        @test m.hot_loop !== nothing && m.hot_loop.counts.fp64_fma ≥ 2      # the two fmas of the loop
        @test m.ir !== nothing && m.ir.total > 0
        VENDOR == "cuda" ? (@test ismissing(m.registers)) : (@test m.registers isa Int)
        irm = kernel_ir_mix(backend, ck)
        @test irm.counts == m.ir.counts
        fl = fp64_issue_floor(m; n_slots = n * 100, peak_fp64_flops = 1e12, kernel_time_s = 1e-3)
        @test fl.floor_s > 0 && !ismissing(fl.fp64_issue_fraction)
        mx = kernel_instruction_mix(backend, ck; target = cross_target, ir = false)
        @test !mx.native && mx.target == cross_target && mx.total > 0 && mx.coverage ≥ 0.98
        @info "instruction mix" native_total = m.total hot_loop = m.hot_loop.total confidence = m.hot_loop_confidence cross_total = mx.total unclassified = m.unclassified_opcodes
    end

    @testset "probes: launch overhead, host snapshot, warm-up" begin
        o = measure_launch_overhead(backend; n = 100)
        @test o.device_s > 0 && o.enqueue_s > 0 && o.roundtrip_s > o.enqueue_s && o.queue_depth ≥ 1
        h = host_snapshot(backend)
        @test h.backend == string(nameof(typeof(backend))) && !ismissing(h.gpu_runtime) && !ismissing(h.gpu_package)
        VENDOR == "cuda" && @test !ismissing(h.gpu_driver) && !ismissing(h.gpu_kernel_module)   # in-tree amdgpu has no sysfs version
        @test !isempty(diagnostics_dict(h; prefix = "host_")["host_gpu_runtime"])
        @info "probes" launch_overhead = sprint(show, o) host = sprint(show, h) versions = (h.gpu_driver, h.gpu_runtime, h.gpu_kernel_module, h.gpu_package)
        timer = LaunchTimer()
        lane = launch_lane(timer, backend)
        for _ in 1:3
            e0 = launch_tick(timer, backend)
            gpudiag_probe_kernel!(backend, 256)(out, x, Int32(1000); ndrange = n)
            launch_tock!(timer, lane, backend, e0)
        end
        @test length(launch_times(timer; skip_first = true)[gpu_device(backend)]) == 2 && first_launch_s(timer)[gpu_device(backend)] > 0
        @test occursin("median_s", sprint(show, MIME"text/plain"(), timer))
    end

    @testset "telemetry sampler" begin
        s = gpu_sample(backend)
        @test s.power_W > 0 && 0 ≤ s.compute_util ≤ 1 && s.vram_used_B > 0
        # #3 columns: clocks, temperature, power limit, throttle bitmask (NaN where the vendor has none)
        @test s.sm_clock_MHz > 0 && s.mem_clock_MHz > 0 && 0 < s.temperature_C < 120 && s.power_limit_W > 0
        if VENDOR == "cuda"
            @test isfinite(s.throttle_reasons) && isinteger(s.throttle_reasons) && throttle_reasons(s.throttle_reasons) isa Vector{Symbol}
        else
            @test isnan(s.throttle_reasons) && isfinite(s.hotspot_C)
        end
        @info "gpu_sample" sample = s decoded_throttle = throttle_reasons(s.throttle_reasons)
        # The child needs a few seconds to start (it loads CUDA.jl for NVML on NVIDIA), so the
        # workload must outlast that: launch until ~8 s of wall time have passed.
        result, telem = with_gpu_sampler(backend, 0.2; counters = :none) do
            t_end = time() + 8
            while time() < t_end
                gpudiag_probe_kernel!(backend, 256)(out, x, Int32(20_000); ndrange = n)
                KernelAbstractions.synchronize(backend)
            end
            :done
        end
        @test result === :done
        @test telem.ticks ≥ 2 && !ismissing(telem.first_sample_s) && haskey(telem, :power_W)
        st = gpu_telemetry_stats(telem)
        @test st["samples"] == telem.ticks && haskey(st, "power_W_mean")
        @test haskey(st, "sm_clock_MHz_busy_median") && haskey(st, "power_capped_fraction") && 0 ≤ st["power_capped_fraction"] ≤ 1
        dd = diagnostics_dict(telem; prefix = "sampler_")
        @test dd["sampler_ticks"] == telem.ticks && haskey(dd, "sampler_sm_clock_MHz_busy_median")
        @test occursin("ticks", sprint(show, MIME"text/plain"(), telem))
        @info "clocks" sm_clock_busy_median = st["sm_clock_MHz_busy_median"] power_capped_fraction = st["power_capped_fraction"] temperature_peak = get(st, "temperature_C_peak", NaN)
        @info "telemetry" ticks = telem.ticks first_sample_s = telem.first_sample_s power_W_mean = st["power_W_mean"] starved = telem.starved
    end
end
