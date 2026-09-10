# ── Cheap vendor-neutral probes: launch overhead, host environment, warm-up ─────────────────
#
# Three small instruments motivated by incidents no other column caught: a kernel diagnosed as
# latency-bound with nothing saying what a launch costs on that device and driver; a rented
# node 14× slower than a smaller local box because Julia started one thread per host core on a
# cgroup that granted a fraction of them; JIT time landing inside the telemetry window. All run
# on every backend (the CPU backend included) and feed the report layer.

# ---- launch overhead ------------------------------------------------------------------------

@kernel function _noop_kernel!(out)
    i = @index(Global, Linear)
    @inbounds out[i] = one(eltype(out))
end

"""
    LaunchOverhead

Result of [`measure_launch_overhead`](@ref), all medians over `n` launches of a one-instruction
kernel, in seconds: `device_s` (device-event time per launch — the smallest kernel the device
can run), `enqueue_s` (host time to enqueue one asynchronous launch), `roundtrip_s` (host time
of one launch followed by a synchronize — the cost of a per-launch sync). `queue_depth` is
`roundtrip_s / enqueue_s`: how many launches the host can queue in the time one takes to
complete, i.e. how far a launch loop can run ahead before it stalls.
"""
struct LaunchOverhead
    n::Int
    device_s::Float64
    enqueue_s::Float64
    roundtrip_s::Float64
    queue_depth::Float64
end

"""
    measure_launch_overhead(backend; n = 200, workgroup = 64) -> LaunchOverhead

Cost of a kernel launch on `backend`: `n` launches of a one-instruction kernel over `workgroup`
work-items, timed three ways (device events per launch, host enqueue time without a sync, host
round-trip with a sync per launch), each reported as the median. Sets the floor for how small a
work item can be before launches dominate; differs by an order of magnitude between drivers,
and between a VM and bare metal. Requires the `:events` capability. Costs well under a second.
"""
function measure_launch_overhead(backend::Backend; n::Integer = 200, workgroup::Integer = 64)
    _require(backend, :events, :measure_launch_overhead)
    n > 0 && workgroup > 0 || throw(ArgumentError("measure_launch_overhead: n and workgroup must be > 0"))
    out = Adapt.adapt(backend, zeros(Float32, Int(workgroup)))
    kern = _noop_kernel!(backend, Int(workgroup))
    kern(out; ndrange = length(out))                        # JIT warm-up
    KernelAbstractions.synchronize(backend)
    # device time per launch, launches queued asynchronously
    dev = Float64[]
    for _ in 1:n
        e0 = gpu_event(backend)
        kern(out; ndrange = length(out))
        e1 = gpu_event(backend)
        push!(dev, gpu_elapsed(e0, e1))
    end
    KernelAbstractions.synchronize(backend)
    # host enqueue time (no sync inside the loop)
    enq = Float64[]
    for _ in 1:n
        t0 = time_ns()
        kern(out; ndrange = length(out))
        push!(enq, (time_ns() - t0) * 1.0e-9)
    end
    KernelAbstractions.synchronize(backend)
    # host round trip with a sync per launch
    rt = Float64[]
    for _ in 1:n
        t0 = time_ns()
        kern(out; ndrange = length(out))
        KernelAbstractions.synchronize(backend)
        push!(rt, (time_ns() - t0) * 1.0e-9)
    end
    med(v) = Float64(Statistics.median(v))
    e, r = med(enq), med(rt)
    return LaunchOverhead(Int(n), med(dev), e, r, e > 0 ? r / e : Inf)
end

function diagnostics_dict(o::LaunchOverhead; prefix::AbstractString = "")
    d = _newdict()
    for k in fieldnames(LaunchOverhead)
        _put!(d, prefix, k, getfield(o, k))
    end
    return d
end
Base.show(io::IO, o::LaunchOverhead) = print(io, "LaunchOverhead(device ", round(o.device_s * 1e6; sigdigits = 3), " µs, enqueue ",
    round(o.enqueue_s * 1e6; sigdigits = 3), " µs, round trip ", round(o.roundtrip_s * 1e6; sigdigits = 3), " µs, queue depth ",
    round(o.queue_depth; sigdigits = 3), "; median of ", o.n, ")")

# ---- host environment snapshot ---------------------------------------------------------------

"""
    backend_versions(backend) -> NamedTuple

Backend hook (optional): the vendor's version strings for [`host_snapshot`](@ref) — any of
`driver`, `runtime`, `package` (the vendor Julia package and its version). Default: empty."""
backend_versions(::Backend) = (;)

"""
    HostSnapshot

Result of [`host_snapshot`](@ref): what the run happened on, for comparing manifests written
months apart and for the host-side traps no device counter shows.

- `julia`, `julia_threads`, `blas_threads`, `os`, `kernel` (OS kernel release), `hostname`;
- `cpu_threads` (what Julia sees) and `cpu_quota` (the cgroup CPU quota in cores, `missing`
  when unlimited or not on Linux) — a pod sees the node's cores while the cgroup grants a
  fraction;
- `memory_total_B`, `memory_available_B`, `memory_limit_B` (cgroup limit, `missing` when none);
- `backend`, `gpu_driver`, `gpu_runtime`, `gpu_kernel_module` (the `nvidia` / `amdgpu` module
  version), `gpu_package` (from the vendor extension), `packages` (versions of GPUDiagnostics
  and KernelAbstractions);
- `warnings::Vector{String}` — findings, as data: Julia threads over the cgroup quota, or
  Julia × BLAS threads over the host cores.
"""
struct HostSnapshot
    julia::String
    julia_threads::Int
    blas_threads::Union{Missing, Int}
    cpu_threads::Int
    cpu_quota::Union{Missing, Float64}
    memory_total_B::Int
    memory_available_B::Int
    memory_limit_B::Union{Missing, Int}
    os::String
    kernel::String
    hostname::String
    backend::String
    gpu_driver::Union{Missing, String}
    gpu_runtime::Union{Missing, String}
    gpu_kernel_module::Union{Missing, String}
    gpu_package::Union{Missing, String}
    packages::Dict{String, String}
    warnings::Vector{String}
end

# cgroup v2 `cpu.max` is "<quota> <period>" or "max <period>"; v1 has cfs_quota_us / cfs_period_us.
function _cpu_quota_from_max(s::AbstractString)
    parts = split(strip(s))
    length(parts) == 2 || return missing
    parts[1] == "max" && return missing
    q = tryparse(Float64, parts[1]); p = tryparse(Float64, parts[2])
    (q === nothing || p === nothing || p <= 0) && return missing
    return q / p
end
_read_or_nothing(path) = isfile(path) ? (try strip(read(path, String)); catch; nothing end) : nothing

# The controller files live at the process's cgroup or at an ancestor (leaf scopes often have no
# controllers enabled); walk up from the leaf and take the first present.
function _cgroup_file(name::AbstractString)
    Sys.islinux() || return nothing
    self = _read_or_nothing("/proc/self/cgroup")
    self === nothing && return nothing
    for line in split(self, '\n')
        f = split(line, ':')
        length(f) == 3 || continue
        rel = f[3]
        while true
            path = joinpath("/sys/fs/cgroup", lstrip(rel, '/'), name)
            v = _read_or_nothing(path)
            v === nothing || return v
            (rel == "/" || isempty(rel)) && break
            rel = dirname(rel)
        end
    end
    return nothing
end
function _cgroup_cpu_quota()
    v = _cgroup_file("cpu.max")
    v === nothing || return _cpu_quota_from_max(v)
    q = _cgroup_file("cpu.cfs_quota_us"); p = _cgroup_file("cpu.cfs_period_us")   # cgroup v1
    (q === nothing || p === nothing) && return missing
    return _cpu_quota_from_max(q * " " * p)
end
function _cgroup_memory_limit()
    v = something(_cgroup_file("memory.max"), _cgroup_file("memory.limit_in_bytes"), "max")
    v == "max" && return missing
    n = tryparse(Int, v)
    (n === nothing || n >= 2^62) && return missing   # v1 reports "unlimited" as a huge number
    return n
end

_module_version(name) = something(_read_or_nothing("/sys/module/$name/version"), missing)

"""
    host_snapshot(backend = nothing) -> HostSnapshot

Snapshot of the host and software the run happens on (see [`HostSnapshot`](@ref)); the vendor
version strings are filled in when a `backend` is given and its extension implements
[`backend_versions`](@ref). Warnings are returned as data, never logged.
"""
function host_snapshot(backend::Union{Nothing, Backend} = nothing)
    blas = try Int(LinearAlgebra.BLAS.get_num_threads()) catch; missing end
    quota = _cgroup_cpu_quota()
    vv = backend === nothing ? (;) : backend_versions(backend)
    modver = backend === nothing ? missing :
        something(_module_version("nvidia"), _module_version("amdgpu"), missing)
    pkgs = Dict{String, String}("GPUDiagnostics" => string(pkgversion(@__MODULE__)),
        "KernelAbstractions" => string(pkgversion(KernelAbstractions)))
    warnings = String[]
    nt = Threads.nthreads()
    if !ismissing(quota) && nt > quota + 0.5
        push!(warnings, "Julia runs $nt threads but the cgroup grants $(round(quota; digits = 2)) CPU cores: oversubscribed")
    end
    if !ismissing(blas) && nt > 1 && blas > 1 && nt * blas > Sys.CPU_THREADS
        push!(warnings, "$nt Julia threads × $blas BLAS threads exceed the $(Sys.CPU_THREADS) host threads")
    end
    kern = something(_read_or_nothing("/proc/sys/kernel/osrelease"), string(Sys.KERNEL))
    return HostSnapshot(string(VERSION), nt, blas, Sys.CPU_THREADS, quota, Int(Sys.total_memory()), Int(Sys.free_memory()),
        _cgroup_memory_limit(), string(Sys.KERNEL), kern, gethostname(),
        backend === nothing ? "" : string(nameof(typeof(backend))),
        haskey(vv, :driver) ? string(vv.driver) : missing, haskey(vv, :runtime) ? string(vv.runtime) : missing, modver,
        haskey(vv, :package) ? string(vv.package) : missing, pkgs, warnings)
end

function diagnostics_dict(h::HostSnapshot; prefix::AbstractString = "")
    d = _newdict()
    for k in fieldnames(HostSnapshot)
        k in (:packages, :warnings) && continue
        _put!(d, prefix, k, getfield(h, k))
    end
    for (k, v) in h.packages
        _put!(d, prefix, "pkg_" * k, v)
    end
    isempty(h.warnings) || _put!(d, prefix, :warnings, join(h.warnings, "; "))
    return d
end
Base.show(io::IO, h::HostSnapshot) = print(io, "HostSnapshot(", h.hostname, ": Julia ", h.julia, ", ", h.julia_threads, " threads on ",
    h.cpu_threads, " CPUs", ismissing(h.cpu_quota) ? "" : " (quota $(round(h.cpu_quota; digits = 1)))",
    isempty(h.backend) ? "" : ", $(h.backend)", isempty(h.warnings) ? "" : ", $(length(h.warnings)) warning(s)", ")")
function Base.show(io::IO, ::MIME"text/plain", h::HostSnapshot)
    println(io, "HostSnapshot: ", h.hostname, " (", h.os, " ", h.kernel, ")")
    rows = (
        "julia" => h.julia, "julia_threads" => h.julia_threads, "blas_threads" => _fmt_missing(h.blas_threads),
        "cpu_threads" => h.cpu_threads, "cpu_quota" => ismissing(h.cpu_quota) ? "— (unlimited)" : string(round(h.cpu_quota; digits = 2)),
        "memory_total_GiB" => round(h.memory_total_B / 2^30; digits = 1),
        "memory_available_GiB" => round(h.memory_available_B / 2^30; digits = 1),
        "memory_limit_GiB" => ismissing(h.memory_limit_B) ? "— (unlimited)" : string(round(h.memory_limit_B / 2^30; digits = 1)),
        "backend" => isempty(h.backend) ? "—" : h.backend,
        "gpu_driver" => _fmt_missing(h.gpu_driver), "gpu_runtime" => _fmt_missing(h.gpu_runtime),
        "gpu_kernel_module" => _fmt_missing(h.gpu_kernel_module), "gpu_package" => _fmt_missing(h.gpu_package),
    )
    for (k, v) in rows
        println(io, "  ", rpad(k, 22), v)
    end
    for (k, v) in sort!(collect(h.packages))
        println(io, "  ", rpad("pkg " * k, 22), v)
    end
    for w in h.warnings
        println(io, "  WARNING: ", w)
    end
    return nothing
end
