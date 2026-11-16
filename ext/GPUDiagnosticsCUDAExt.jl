module GPUDiagnosticsCUDAExt

# CUDA.jl implementations of the vendor-GPU API declared in src/device_api.jl. Loaded
# automatically when both GPUDiagnostics and CUDA are in the session. Telemetry
# (power/utilization/memory, GPM counters) goes through NVML; device props through CUDA attributes.

using GPUDiagnostics
using CUDA
using CUDA: NVML
import GPUCompiler
import LLVM

const GD = GPUDiagnostics
GD.backend_counter_collector(::CUDABackend) = NsightCompute()

# NVML handle for the current CUDA device (NVML indexes by UUID, not the CUDA ordinal).
_nvml() = NVML.Device(CUDA.uuid(CUDA.device()))

# Counter access is a per-device runtime question; these capabilities describe the API.
# GPM counters depend on the device and driver
# (Hopper+ and recent consumer drivers); the FEATURE is the sampler knowing how to ask.
for f in (:devices, :device_props, :events, :telemetry, :telemetry_counters, :peak_flops, :fp64,
        :kernel_inventory, :resources, :occupancy, :native_mix, :ir_mix, :hw_counters)
    @eval GD.supports(::CUDABackend, ::Val{$(QuoteNode(f))}) = true
end

GD.gpu_device_count(::CUDABackend) = Int(length(CUDA.devices()))
GD.gpu_device(::CUDABackend) = Int(CUDA.deviceid(CUDA.device())) + 1          # 0-based CUDA → 1-based API
function GD.gpu_device!(::CUDABackend, i::Integer)
    prev = CUDA.deviceid(CUDA.device()) + 1
    CUDA.device!(i - 1)
    return prev
end
GD.gpu_name(::CUDABackend) = CUDA.name(CUDA.device())
GD.gpu_power(::CUDABackend) = NVML.power_usage(_nvml())                  # Watts (Float64)
GD.gpu_utilization(::CUDABackend) = NVML.utilization_rates(_nvml())     # (compute, memory) ∈ [0,1]
GD.gpu_memory_info(::CUDABackend) = NVML.memory_info(_nvml())           # (total, free, used) bytes

# Device-event timing on the task-local stream (the one KernelAbstractions launches on). CuEvent's
# default flags keep timing enabled; `elapsed` needs the stop event complete → synchronize it.
GD.gpu_event(::CUDABackend) = (e = CUDA.CuEvent(); CUDA.record(e, CUDA.stream()); e)
function GD.gpu_elapsed(start::CUDA.CuEvent, stop::CUDA.CuEvent)
    CUDA.synchronize(stop)
    return Float64(CUDA.elapsed(start, stop))   # seconds
end
GD.gpu_sm_count(::CUDABackend) =
    Int(CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT))   # attribute() is Int32
GD.gpu_max_threads_per_sm(::CUDABackend) =
    Int(CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR))

GD.gpu_arch(::CUDABackend) = (cc = CUDA.capability(CUDA.device()); "$(cc.major).$(cc.minor)")

# ── Telemetry sources (src/sampler.jl hooks) ─────────────────────────────────────────────────
# The CUDA runtime is touched only HERE, in the parent, to map our ordinals to NVML uuids (stable
# under CUDA_VISIBLE_DEVICES). The sampler child loads CUDA.jl for its NVML bindings only (this
# extension then provides the `nvml` source kind there too), never creates a CUDA context, and is
# immune to this process's CUDA locks and Julia's GC/timer coupling.
function GD.gpu_sampler_sources(::CUDABackend, device_ids::AbstractVector{<:Integer}, counters::Symbol)
    cudevs = collect(CUDA.devices())
    specs = ["nvml:GPU-$(CUDA.uuid(cudevs[i]))=$i" for i in device_ids]
    return (specs = specs, packages = [Base.PkgId(CUDA)])
end

# NVML per-device source: power / utilization / memory / clocks / temperature / power limit /
# throttle bitmask via the plain NVML queries, plus — when
# the device supports GPU Performance Monitoring (Hopper and newer, incl. consumer Blackwell on
# recent drivers; no admin privileges) and counters are wanted — the GPM metrics. GPM is
# interval-based: a metric is the difference of two samples, so the source keeps two sample
# buffers and evaluates each tick against the previous tick's sample (rows cover exactly the
# preceding `dt`; the priming call takes a 100 ms interval).
const _GPM_METRICS = (   # column => (nvmlGpmMetricId_t, unit scale: percent → fraction; MiB/s as is)
    (:sm_util, 2, 0.01), (:sm_occupancy, 3, 0.01), (:fp64_util, 11, 0.01), (:dram_bw_util, 10, 0.01),
    (:fp32_util, 12, 0.01), (:fp16_util, 13, 0.01), (:tensor_util, 5, 0.01), (:int_util, 4, 0.01),
    (:pcie_tx_MiBps, 20, 1.0), (:pcie_rx_MiBps, 21, 1.0), (:nvlink_rx_MiBps, 60, 1.0), (:nvlink_tx_MiBps, 61, 1.0),
)
const _GPM_NAMES = Tuple(m[1] for m in _GPM_METRICS)

mutable struct NVMLSource <: GD.SamplerSource
    device::Int
    dev::NVML.Device
    gpm::Bool
    samples::Vector{NVML.nvmlGpmSample_t}   # two buffers, swapped every tick
    older::Int
    primed::Bool
    buf::Vector{UInt8}                      # nvmlGpmMetricsGet_t scratch (~19 kB, 477-entry metric array)
end

function _gpm_supported(dev::NVML.Device)
    try
        sup = Ref(NVML.nvmlGpmSupport_t(NVML.NVML_GPM_SUPPORT_VERSION, 0))
        NVML.nvmlGpmQueryDeviceSupport(dev, sup)
        return sup[].isSupportedDevice != 0
    catch err   # NVML_ERROR_NOT_SUPPORTED / FUNCTION_NOT_FOUND on old drivers ⇒ simply no GPM
        @debug "GPM support query failed" exception = err
        return false
    end
end

function GD._sampler_source(::Val{:nvml}, rest::AbstractString, counters::Symbol)
    m = match(r"^GPU-([0-9a-fA-F-]+)=(\d+)$", rest)
    m === nothing && throw(ArgumentError("nvml source spec must be GPU-<uuid>=<device>, got $rest"))
    dev = NVML.Device(Base.UUID(m.captures[1]))
    gpm = counters != :none && _gpm_supported(dev)
    samples = NVML.nvmlGpmSample_t[]
    if gpm
        for _ in 1:2
            s = Ref{NVML.nvmlGpmSample_t}()
            NVML.nvmlGpmSampleAlloc(s)
            push!(samples, s[])
        end
    end
    return NVMLSource(parse(Int, m.captures[2]), dev, gpm, samples, 1, false,
        gpm ? zeros(UInt8, sizeof(NVML.nvmlGpmMetricsGet_t)) : UInt8[])
end

# Guarded NVML query: an unsupported / failing query is a NaN in the row, never an exception.
function _nvml_or_nan(f, what)
    try
        return Float64(f())
    catch err
        @debug "NVML $what query failed" exception = err
        return NaN
    end
end
_nvml_clock(dev, type) = (r = Ref{Cuint}(); NVML.nvmlDeviceGetClockInfo(dev, type, r); r[])   # MHz
_nvml_power_limit(dev) = (r = Ref{Cuint}(); NVML.nvmlDeviceGetEnforcedPowerLimit(dev, r); r[] / 1000)   # mW → W
# The bitmask (nvmlClocksEventReason* bits); the entry point was renamed in NVML 12.2.
function _nvml_throttle(dev)
    r = Ref{Culonglong}(0)
    if NVML.version() >= v"12.2"
        NVML.nvmlDeviceGetCurrentClocksEventReasons(dev, r)
    else
        NVML.nvmlDeviceGetCurrentClocksThrottleReasons(dev, r)
    end
    return r[]
end

function GD.sample!(s::NVMLSource)
    ur = NVML.utilization_rates(s.dev)
    base = (power_W = Float64(NVML.power_usage(s.dev)), compute_util = Float64(ur.compute),
        mem_util = Float64(ur.memory), vram_used_B = Float64(NVML.memory_info(s.dev).used),
        sm_clock_MHz = _nvml_or_nan(() -> _nvml_clock(s.dev, NVML.NVML_CLOCK_SM), "SM clock"),
        mem_clock_MHz = _nvml_or_nan(() -> _nvml_clock(s.dev, NVML.NVML_CLOCK_MEM), "memory clock"),
        temperature_C = _nvml_or_nan(() -> NVML.temperature(s.dev, NVML.NVML_TEMPERATURE_GPU), "temperature"),
        hotspot_C = NaN,   # NVML exposes no junction sensor (nvmlTemperatureSensors_t is GPU only)
        power_limit_W = _nvml_or_nan(() -> _nvml_power_limit(s.dev), "power limit"),
        throttle_reasons = _nvml_or_nan(() -> _nvml_throttle(s.dev), "throttle reasons"))
    s.gpm || return base
    if !s.primed
        NVML.nvmlGpmSampleGet(s.dev, s.samples[s.older])
        s.primed = true
        sleep(0.1)
    end
    newer = 3 - s.older
    NVML.nvmlGpmSampleGet(s.dev, s.samples[newer])
    vals = _gpm_metrics!(s.buf, s.samples[s.older], s.samples[newer])
    s.older = newer
    return merge(base, NamedTuple{_GPM_NAMES}(Tuple(vals)))
end

function GD.close!(s::NVMLSource)
    for smp in s.samples
        try
            NVML.nvmlGpmSampleFree(smp)
        catch
        end
    end
    empty!(s.samples)
end

# nvmlGpmMetricsGet_t is a ~19 kB struct whose metric array the bindings expose as an opaque
# NTuple; drive it through a byte buffer and field offsets. Per-metric status lands in each
# entry's nvmlReturn (unsupported ⇒ nan); NVML_GPM_METRICS_GET_VERSION is literally 1.
const _GetT = NVML.nvmlGpmMetricsGet_t
const _MetricT = NVML.nvmlGpmMetric_t
function _gpm_metrics!(buf::Vector{UInt8}, older::NVML.nvmlGpmSample_t, newer::NVML.nvmlGpmSample_t)
    fill!(buf, 0)
    vals = fill(NaN, length(_GPM_METRICS))
    GC.@preserve buf begin
        p = pointer(buf)
        unsafe_store!(Ptr{Cuint}(p + fieldoffset(_GetT, 1)), Cuint(NVML.NVML_GPM_METRICS_GET_VERSION))
        unsafe_store!(Ptr{Cuint}(p + fieldoffset(_GetT, 2)), Cuint(length(_GPM_METRICS)))
        unsafe_store!(Ptr{NVML.nvmlGpmSample_t}(p + fieldoffset(_GetT, 3)), older)
        unsafe_store!(Ptr{NVML.nvmlGpmSample_t}(p + fieldoffset(_GetT, 4)), newer)
        mp(i) = Ptr{_MetricT}(p + fieldoffset(_GetT, 5) + (i - 1) * sizeof(_MetricT))
        for (i, m) in enumerate(_GPM_METRICS)
            unsafe_store!(mp(i).metricId, Cuint(m[2]))
        end
        try
            NVML.nvmlGpmMetricsGet(Ptr{_GetT}(p))
            for (i, m) in enumerate(_GPM_METRICS)
                unsafe_load(mp(i).nvmlReturn) == NVML.NVML_SUCCESS || continue
                v = unsafe_load(mp(i).value)
                isfinite(v) && (vals[i] = v * m[3])
            end
        catch err
            @debug "nvmlGpmMetricsGet failed" exception = err
        end
    end
    return vals
end

# ── Compile-time resource report (src/resources.jl hooks) ───────────────────────────────────
# Recording = GPUCompiler's compile hook around `f`: CUDACore reports every kernel lookup to it
# (`compile_or_lookup`, under `cufunction_lock`), with the active context. Afterwards the
# compiled results of each job are read back from GPUCompiler's cache, and the `CuFunction` the
# runtime linked for that context is taken from them: nothing is compiled or linked here.
# Attributes come from `cuFuncGetAttribute` on that function, occupancy from the driver's
# `cuOccupancyMaxActiveBlocksPerMultiprocessor`, capacities from the CURRENT device.
const _CC = CUDA.CUDACore

"""
    RecordedKernel{F, TT}

A kernel recorded by `compiled_kernels(f, ::CUDABackend)`: the `job` CUDA.jl compiled for the
Julia function type `F` and argument types `TT`, and the `CuFunction` `fun` it linked.
"""
struct RecordedKernel{F, TT}
    job::GPUCompiler.CompilerJob
    fun::CUDA.CuFunction
end

function _job_signature(job)
    sig = Base.unwrap_unionall(job.source.specTypes)
    return sig.parameters[1], Tuple{sig.parameters[2:end]...}
end

function GD.backend_record_kernels(f, ::CUDABackend)
    seen = Tuple{GPUCompiler.CompilerJob, CUDA.CuContext}[]
    lk = ReentrantLock()
    hook(job) = job.config.kernel && Base.@lock lk begin
        key = (job, CUDA.context())
        key in seen || push!(seen, key)
    end
    GPUCompiler.with_compile_hook(f, hook)
    out = GD.CompiledKernel[]
    Base.@lock _CC.cufunction_lock begin
        for (job, ctx) in seen
            res = GPUCompiler.cached_results(_CC.CUDACompilerResults, job)
            res === nothing && continue
            i = findfirst(e -> e[1] === ctx, res.kernels)
            i === nothing && continue      # looked up but never linked here (reflection only)
            F, TT = _job_signature(job)
            push!(out, GD.backend_wrap_kernel(RecordedKernel{F, TT}(job, res.kernels[i][2])))
        end
    end
    return out
end

_fattr(k::RecordedKernel, a) = Int(CUDA.attributes(k.fun)[a])

function GD.backend_kernel_attributes(::CUDABackend, k::RecordedKernel)
    return (;
        registers = _fattr(k, CUDA.FUNC_ATTRIBUTE_NUM_REGS),
        local_mem_bytes = _fattr(k, CUDA.FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES),
        shared_mem_bytes = _fattr(k, CUDA.FUNC_ATTRIBUTE_SHARED_SIZE_BYTES),
        const_mem_bytes = _fattr(k, CUDA.FUNC_ATTRIBUTE_CONST_SIZE_BYTES),
        max_threads_per_block = _fattr(k, CUDA.FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK),
    )
end

function GD.backend_kernel_occupancy(::CUDABackend, k::RecordedKernel, block_size::Int)
    dev = CUDA.device()
    return (;
        active_blocks_per_sm = Int(CUDA.active_blocks(k.fun, block_size)),
        warp_size = Int(CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_WARP_SIZE)),
        max_threads_per_sm = Int(CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)),
        shared_mem_per_sm = Int(CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)),
    )
end

# PTX / SASS binary versions the kernel was built for, plus the `ptxas --verbose` report of a
# regenerated compile. The driver attribute `local_mem_bytes` lumps the call-ABI stack frame
# (arguments/returns of device functions Julia did not inline, private arrays) together with
# true register spills; only ptxas separates them, and it also names every function it
# assembled out of line. CUDA.jl ships ptxas (CUDA_Compiler_jll) and compiles through it, so
# the same binary is run here on the module PTX of the recorded job (the options the runtime
# compiled with: `always_inline`, `maxthreads` = the static KA workgroup size); the register
# count is checked against the runtime attribute so a mismatched regeneration is flagged, not
# trusted. Costs a few seconds of compiler time; nothing is launched.
function GD.backend_kernel_isa_info(::CUDABackend, ck::GD.CompiledKernel{<:RecordedKernel})
    k = ck.kernel
    info = Dict{String, Any}()
    vnum(x) = VersionNumber(divrem(x, 10)...)
    info["ptx_version"] = string(vnum(_fattr(k, CUDA.FUNC_ATTRIBUTE_PTX_VERSION)))
    info["binary_version"] = string(vnum(_fattr(k, CUDA.FUNC_ATTRIBUTE_BINARY_VERSION)))
    try
        merge!(info, _ptxas_report(k.job))
        if haskey(info, "ptxas_registers")
            info["ptxas_matches_attributes"] = info["ptxas_registers"] == _fattr(k, CUDA.FUNC_ATTRIBUTE_NUM_REGS)
        end
    catch err
        @warn "kernel_resources: ptxas report unavailable — reporting driver attributes only" exception = err
    end
    return info
end

_ptxas_cmd() = _CC.CUDA_Compiler_jll.ptxas()

function _ptxas_report(job)
    io = IOBuffer()
    GPUCompiler.code_native(io, job; raw = true, dump_module = true)   # the module PTX of the recorded job
    ptx = String(take!(io))
    m = match(r"\.target\s+(sm_\w+)", ptx)
    m === nothing && error("no .target line in the generated PTX")
    arch = m[1]
    ptxfile = tempname(; cleanup = false) * ".ptx"
    write(ptxfile, ptx)
    try
        cmd = `$(_ptxas_cmd()) --verbose --gpu-name $arch --output-file /dev/null $ptxfile`
        buf = IOBuffer()   # ptxas writes its report to stderr; capture both streams
        run(pipeline(ignorestatus(cmd); stdout = buf, stderr = buf))
        return GD._parse_ptxas_verbose(String(take!(buf)))
    finally
        rm(ptxfile; force = true)
    end
end


# ── Static instruction mix (src/instruction_mix.jl hook) ────────────────────────────────────
# CUDA.jl's `code_sass` disassembles the cubin from a CUPTI module-load callback, i.e. it
# LOADS the module on the current device — impossible for a cubin of another architecture.
# So the job is compiled here directly (`CUDACore.compile`: LLVM → PTX → the bundled ptxas →
# cubin, no device involved) and the bundled `nvdisasm` reads the cubin. The job is the
# recorded one (the runtime's own config, `sm_NNa` for the device it ran on) or, for
# `target = "sm_90"`, the same options with `arch` overridden; ptxas's `.sectioninfo @"SHI_REGISTERS=N"` gives the
# register count to check against the runtime attribute.
const _GPUC = GPUCompiler

# The CompilerJob of `ck` (the recorded one; for a target, its options with `arch` overridden) and its ISA name.
function _mix_job(ck::GD.CompiledKernel{<:RecordedKernel}, target)
    job = ck.kernel.job
    if target !== nothing
        kw = ck.workgroup_size === nothing ? (;) : (; maxthreads = Int(ck.workgroup_size))
        config = _CC.compiler_config(CUDA.device(); kernel = true, always_inline = job.config.always_inline,
            kw..., arch = String(target))
        job = _GPUC.CompilerJob(job.source, config)
    end
    return job, _CC.cpu_name(job.config.params.sm)
end

function GD.backend_kernel_machine_code(::CUDABackend, ck::GD.CompiledKernel{<:RecordedKernel}, target)
    job, isa = _mix_job(ck, target)
    compiled = _CC.compile(job)
    cubin = tempname(; cleanup = false) * ".cubin"
    write(cubin, compiled.image)
    text = try
        read(`$(_CC.CUDA_Compiler.nvdisasm()) --print-code $cubin`, String)
    finally
        rm(cubin; force = true)
    end
    # nvdisasm prints no register count; kernel_resources carries ptxas's
    return (; text, vendor = :nvidia, isa, native = target === nothing, registers = missing)
end

# Typed IR count: the optimized module of the same job, walked with CUDACore's LLVM.jl.
function GD.backend_kernel_ir_counts(::CUDABackend, ck::GD.CompiledKernel{<:RecordedKernel}, target)
    job, isa = _mix_job(ck, target)
    functions = _GPUC.JuliaContext() do ctx
        ir, _ = _GPUC.compile(:llvm, job)
        GD._ir_counts(LLVM, ir)
    end
    return (; functions, isa)
end


# Versions for host_snapshot: the CUDA driver and runtime CUDA.jl binds, the NVML driver string
# when NVML answers, and the package version.
function GD.backend_versions(::CUDABackend)
    nvml = try string(NVML.driver_version()) catch; missing end
    return (; driver = string(CUDA.driver_version()) * (ismissing(nvml) ? "" : " (driver $nvml)"),
        runtime = string(CUDA.runtime_version()), package = "CUDA.jl " * string(pkgversion(CUDA)))
end

end
