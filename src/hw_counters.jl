"""
    CounterSet

A profiler preset: native `metrics`, the `validated_on` architecture/tool descriptions,
`passes` (known only in that validation context), and `notes`. Presets express intent;
they do not promise identical counters or collection cost on different architectures.
"""
struct CounterSet
    metrics::Vector{String}
    validated_on::Vector{String}
    passes::Union{Missing, Int}
    notes::String
end

"""
    HWDispatch

Identity and metadata for one logical profiled launch. `id` is tool-local, so identify a
dispatch with `(process_id, device_id, context_id, queue_id, id)`. Numeric IDs retain the
profiler's indexing. `start_s` is in the tool's clock domain, not a host epoch. `duration_s`
is a profiled duration, potentially assembled over replay passes. `resources` and `device`
contain native metadata for THIS dispatch. Unavailable fields are `missing`; `slots` is
the caller's work-item iteration count for this launch, not inferred from the grid.
"""
struct HWDispatch
    id::Int
    process_id::Union{Missing, Int}
    device_id::Union{Missing, Int}
    context_id::Union{Missing, Int}
    queue_id::Union{Missing, Int}
    kernel::String
    start_s::Union{Missing, Float64}
    duration_s::Union{Missing, Float64}
    resources::Dict{String, Any}
    device::Dict{String, Any}
    slots::Union{Missing, Float64}
end

"""
    HWCounters

Portable per-dispatch counter data for one selected kernel. `dispatches` aligns with rows
of `values`; `counters` names its columns, and `units` retains each native metric's unit
(missing where the tool does not supply one). Index by native name: `hc["SQ_WAVES"]`.
`vendor`, `tool`, `name`, `kernel`, and `provenance` describe the collection. Parsers never
load a GPU package, run a profiler, or silently summarize launches. Use
[`hw_counter_derived`](@ref) for derived columns and [`hw_counter_summary`](@ref) for reduction.
"""
struct HWCounters
    vendor::Symbol
    tool::Symbol
    name::String
    kernel::String
    dispatches::Vector{HWDispatch}
    counters::Vector{String}
    values::Matrix{Union{Missing, Float64}}
    units::Dict{String, Union{Missing, String}}
    provenance::Dict{String, Any}
end
Base.length(h::HWCounters) = length(h.dispatches)
Base.keys(h::HWCounters) = h.counters
Base.haskey(h::HWCounters, key::AbstractString) = key in h.counters
function Base.getindex(h::HWCounters, key::AbstractString)
    j = findfirst(==(key), h.counters)
    j === nothing && throw(KeyError(key))
    return h.values[:, j]
end
Base.show(io::IO, h::HWCounters) = print(io, "HWCounters(", h.tool, ": ", length(h),
    " dispatches of ", first(split(h.kernel, '(')), " × ", length(h.counters), " counters)")

"""
    HWCounterAvailability

Tool discovery, separate from permission to collect: `tool`, `executable` (path or
`missing`), `available::Bool`, and `permitted::Union{Missing,Bool}`. A cheap discovery
check leaves `permitted = missing`; only an actual profiling run establishes access.
"""
struct HWCounterAvailability
    tool::Symbol
    executable::Union{Missing, String}
    available::Bool
    permitted::Union{Missing, Bool}
end

_counter_vendor(v::Symbol) = v in (:amd, :nvidia) ? v : throw(ArgumentError("unknown counter vendor :$v"))

"""
    HWCounterCollector

External profiler selected by a backend extension. Use [`RocprofV3`](@ref) or
[`NsightCompute`](@ref) explicitly when preparing an external workload without a GPU runtime.
"""
abstract type HWCounterCollector end

"""    backend_counter_collector(backend) -> HWCounterCollector

Backend hook (`:hw_counters`): select the external profiler used by collection commands
and tool discovery. The CUDA extension returns `NsightCompute()` and the AMDGPU extension
returns `RocprofV3()`. CSV parsing does not use this hook or load an extension.
"""
backend_counter_collector(b::Backend) = throw(BackendUnsupported(b, :hw_counters, :hw_counter_command))

"""    hw_counter_status(backend; executable = nothing) -> HWCounterAvailability
    hw_counter_status(collector::HWCounterCollector; executable = nothing)

Discover the profiler executable without initializing a GPU or asserting profiling
permissions. An unsupported backend reports unavailable. A supplied path can locate a
tool outside PATH. This does not run the tool.
"""
function hw_counter_status(b::Backend; kwargs...)
    supports(b, :hw_counters) || return HWCounterAvailability(:none, missing, false, missing)
    return hw_counter_status(backend_counter_collector(b); kwargs...)
end
function hw_counter_status(collector::HWCounterCollector; executable = nothing)
    tool = _counter_tool(collector)
    path = Sys.which(executable === nothing ? String(tool) : String(executable))
    return HWCounterAvailability(tool, something(path, missing), path !== nothing, missing)
end

"""    hw_counters_available(backend; executable = nothing) -> Bool
    hw_counters_available(collector::HWCounterCollector; executable = nothing)

Whether the backend has a collector and its tool is discoverable. This is NOT a permission
check; [`hw_counter_status`](@ref) leaves permission unknown and does not run the collector.
"""
hw_counters_available(v::Union{Backend, HWCounterCollector}; kwargs...) = hw_counter_status(v; kwargs...).available

"""
    hw_counter_command(backend, cmd; set = :issue, dir, name,
                       metrics = nothing, timeout_s = 600, kill_after_s = 20, ...)
    hw_counter_command(collector::HWCounterCollector, cmd; kwargs...)

Build a `Cmd` wrapping a workload in rocprofv3 or Nsight Compute. Nothing is run or written
by this function. The caller creates `dir` and runs the command. Environment and working
directory are preserved. `metrics` overrides the named preset. `executable` overrides the
tool location. A finite timeout uses the external `timeout` utility; `nothing` disables it.
The backend extension selects the collector. For an external workload, explicitly pass
`RocprofV3()` or `NsightCompute()` without loading AMDGPU or CUDA. Vendor symbols are
used only for offline parsing and preset lookup, not collector selection.

AMD: `kernel_trace=true`, optional `kernel` regex filter. NVIDIA: `kernel`, `launch_skip=0`,
`launch_count=nothing`, `clock_control=:none`, `cache_control=:all`, `replay_mode=:kernel`.
NVIDIA writes `<name>_ncu.csv` and `<name>.ncu-rep`; AMD writes `<name>_counter_collection.csv`
and its companion files. Preset metric availability and pass counts depend on architecture.
Profiling can replay/serialize work and change cache state; profiled times are not ordinary
execution times. Filter and warm up a small workload explicitly.
"""
function hw_counter_command(b::Backend, cmd::Cmd; kwargs...)
    _require(b, :hw_counters, :hw_counter_command)
    return hw_counter_command(backend_counter_collector(b), cmd; kwargs...)
end
function hw_counter_command(collector::HWCounterCollector, cmd::Cmd; set::Symbol = :issue, dir::AbstractString,
        name::AbstractString, metrics = nothing, executable = nothing,
        timeout_s::Union{Nothing, Real} = 600, kill_after_s::Real = 20,
        kernel = nothing, kwargs...)
    vendor = _counter_vendor(collector)
    isempty(name) && throw(ArgumentError("name must not be empty"))
    basename(name) == name && name ∉ (".", "..") || throw(ArgumentError("name must be a file prefix, not a path"))
    timeout_s === nothing || (isfinite(timeout_s) && timeout_s > 0) || throw(ArgumentError("timeout_s must be positive and finite"))
    isfinite(kill_after_s) && kill_after_s > 0 || throw(ArgumentError("kill_after_s must be positive and finite"))
    if metrics === nothing
        haskey(COUNTER_SETS[vendor], set) || throw(ArgumentError("unknown $vendor counter set :$set"))
        metrics = COUNTER_SETS[vendor][set].metrics
    end
    metrics isa AbstractVector || throw(ArgumentError("metrics must be a vector of native names"))
    pmc = String.(metrics)
    isempty(pmc) || any(isempty, pmc) ? throw(ArgumentError("metrics must not be empty")) : nothing
    exe = executable === nothing ? String(_counter_tool(collector)) : String(executable)
    filter = kernel === nothing ? nothing : kernel isa Regex ? kernel.pattern : String(kernel)
    args = _counter_command_args(collector, exe, pmc; dir, name, kernel = filter, kwargs...)
    append!(args, cmd.exec)
    timeout_s === nothing || (args = ["timeout", "-k", string(kill_after_s), string(timeout_s), args...])
    return Cmd(Cmd(args); env = cmd.env, dir = cmd.dir)
end

"""    hw_counter_export_command(report; output, executable = "ncu") -> Cmd

Explicitly export a `.ncu-rep` to raw CSV with the system tool. This returns a command;
[`hw_counters`](@ref) itself never invokes external software. Output is overwritten by ncu.
"""
function hw_counter_export_command(report::AbstractString; output::AbstractString, executable::AbstractString = "ncu")
    return `$executable --import $report --csv --page raw --print-units base --log-file $output`
end

"""
    hw_counters(path; vendor = nothing, name = nothing, kernel = nothing, slots = nothing,
                device_overrides = Dict(), provenance = Dict()) -> HWCounters
    hw_counters(vendor::Symbol, path; kwargs...)

Parse raw CSV, without GPU packages or system tools. `path` is an AMD collection directory
or counter CSV, or an NVIDIA raw CSV/directory. Detection must resolve exactly one collection;
use `vendor` and/or `name` to resolve ambiguity. `kernel` is a substring or regex selecting
ONE full kernel name. Without it, exactly one user kernel must be present.

`slots` is a positive scalar for all selected dispatches, or a vector aligned with them.
`device_overrides` maps native device IDs to dictionaries of known properties, e.g.
`Dict(1 => Dict("n_xcd" => 8, "wave_size" => 64))`. Unknown properties stay missing.
`required_metrics` optionally requires finite values of each listed counter on every
selected dispatch; use the preset's `.metrics` to detect silently omitted tool metrics.
`provenance` can record `tool_version`, `clock_control`, `cache_control`, `replay_mode`,
and `counter_set`; only allowlisted measurement fields are retained, never commands or paths.
"""
function hw_counters(path::AbstractString; vendor = nothing, name = nothing, kwargs...)
    vendor === nothing || _counter_vendor(vendor)
    candidates = Tuple{Symbol, String}[]
    files = isdir(path) ? readdir(path; join = true) : isfile(path) ? [String(path)] :
        throw(ArgumentError("counter path does not exist: $path"))
    for file in files
        endswith(file, ".csv") || continue
        toolvendor = _counter_csv_vendor(file)
        toolvendor === nothing && continue
        vendor === nothing || vendor === toolvendor || continue
        prefix = _collection_name(file, toolvendor)
        name === nothing || prefix == name || continue
        push!(candidates, (toolvendor, file))
    end
    length(candidates) == 1 || throw(ArgumentError("expected one counter collection, found $(length(candidates)); select vendor/name or pass a CSV file"))
    v, file = only(candidates)
    return v === :amd ? _parse_rocprof(file; kwargs...) : _parse_ncu(file; kwargs...)
end
hw_counters(v::Symbol, path::AbstractString; kwargs...) = hw_counters(path; vendor = v, kwargs...)

_collection_name(file, v) = chopsuffix(chopsuffix(basename(file), v === :amd ? "_counter_collection.csv" : "_ncu.csv"), ".csv")
function _counter_csv_vendor(file)
    for line in eachline(file)
        startswith(strip(line), "==") && continue
        f = _csv_fields(line)
        "Dispatch_Id" in f && "Counter_Name" in f && return :amd
        "ID" in f && "Kernel Name" in f && ("Metric Name" in f || any(c -> occursin("__", c), f)) && return :nvidia
    end
    return nothing
end

const _COUNTER_PROVENANCE = ("tool_version", "clock_control", "cache_control", "replay_mode", "counter_set", "passes")
function _counter_provenance(input)
    out = Dict{String, Any}()
    for (k, v) in pairs(input)
        String(k) in _COUNTER_PROVENANCE || continue
        ismissing(v) && continue
        v isa Union{AbstractString, Symbol, Real} || throw(ArgumentError("provenance values must be scalars"))
        out[String(k)] = v isa Symbol ? String(v) : v
    end
    return out
end
_kernel_matches(k::Regex, name) = occursin(k, name)
_kernel_matches(k::AbstractString, name) = occursin(k, name)
_kernel_matches(::Nothing, name) = !startswith(name, "__amd_rocclr_")
function _select_dispatches(dispatches, kernel)
    ids = findall(d -> _kernel_matches(kernel, d.kernel), dispatches)
    names = unique(d.kernel for d in dispatches[ids])
    length(names) == 1 || throw(ArgumentError("kernel selection matched $(length(names)) names; specify kernel (available: $(join(unique(d.kernel for d in dispatches), ", ")))"))
    return ids, only(names)
end
function _counter_slots(slots, n)
    slots === nothing && return fill(missing, n)
    v = slots isa Real ? fill(slots, n) : collect(slots)
    length(v) == n || throw(ArgumentError("slots must align with the selected dispatches"))
    all(x -> x isa Real && isfinite(x) && x > 0, v) || throw(ArgumentError("slots must be positive and finite"))
    return Float64.(v)
end
function _with_slots(d::HWDispatch, slots)
    return HWDispatch(d.id, d.process_id, d.device_id, d.context_id, d.queue_id, d.kernel,
        d.start_s, d.duration_s, d.resources, d.device, slots)
end
function _build_counters(collector::HWCounterCollector, file, dispatches, rows, units; kernel, slots, provenance, required_metrics)
    vendor = _counter_vendor(collector)
    ids, name = _select_dispatches(dispatches, kernel)
    ds = [_with_slots(d, s) for (d, s) in zip(dispatches[ids], _counter_slots(slots, length(ids)))]
    counters = sort!(unique(String[k for i in ids for k in keys(rows[i])]))
    values = Matrix{Union{Missing, Float64}}(missing, length(ids), length(counters))
    for (i, src) in enumerate(ids), (j, c) in enumerate(counters)
        values[i, j] = get(rows[src], c, missing)
    end
    if required_metrics !== nothing
        required_metrics isa AbstractVector || throw(ArgumentError("required_metrics must be a vector of native names"))
        for c in required_metrics
            j = findfirst(==(c), counters)
            (j === nothing || any(ismissing, values[:, j])) &&
                throw(ArgumentError("required counter $c is unavailable on one or more selected dispatches"))
        end
    end
    return HWCounters(vendor, _counter_tool(collector), _collection_name(file, vendor), name, ds,
        counters, values, Dict(c => get(units, c, missing) for c in counters), _counter_provenance(provenance))
end

_safe_ratio(a, b) = ismissing(a) || ismissing(b) || b <= 0 ? missing : a / b
_finite_metric(x) = ismissing(x) || !isfinite(x) ? missing : Float64(x)

"""
    hw_counter_derived(hc) -> Dict{String, Vector{Union{Missing,Float64}}}

Derived columns aligned with `hc.dispatches`, never medians. Unavailable inputs, unsupported
architecture-specific normalization, and zero denominators produce `missing`. Keys whose
inputs were not collected are omitted. Common names are used only for defined comparable
quantities; vendor-specific estimates keep a vendor prefix. See the hardware-counter guide
for denominators and units. Summary reduction is explicit in [`hw_counter_summary`](@ref).
"""
function hw_counter_derived(h::HWCounters)
    out = Dict{String, Vector{Union{Missing, Float64}}}()
    for (i, d) in enumerate(h.dispatches)
        raw = Dict(c => h.values[i, j] for (j, c) in enumerate(h.counters))
        derived = h.vendor === :amd ? _amd_derived(raw, d) : _nvidia_derived(raw, d, h.units)
        for (k, v) in derived
            col = get!(() -> Vector{Union{Missing, Float64}}(missing, length(h)), out, k)
            col[i] = _finite_metric(v)
        end
    end
    return out
end

function _counter_stats!(out, key, values)
    v = Float64[x for x in values if !ismissing(x) && isfinite(x)]
    isempty(v) && return
    m = Float64(Statistics.median(v))
    out[key * "_median"] = m
    out[key * "_min"] = minimum(v)
    out[key * "_max"] = maximum(v)
    out[key * "_samples"] = length(v)
    m == 0 || (out[key * "_rel_spread"] = (maximum(v) - minimum(v)) / abs(m))
    return
end

"""
    hw_counter_summary(hc) -> Dict{String,Any}

Flat reduction with vendor/tool identity, the named device and resource properties common
to every selected dispatch (`device_n_cu`, `resources_registers`, …; the vendor-namespaced
passthrough entries such as `launch__*` and `device__attribute_*` stay on the dispatch), and
`<metric>_median/min/max/samples/rel_spread`. Native counters have `raw_` prefixes and
unit keys; derived columns have `derived_` prefixes. A total profiled duration is emitted
only when every dispatch has a duration. Different device/launch properties are not
silently represented by the first dispatch. Missing keys are omitted.
"""
function hw_counter_summary(h::HWCounters)
    out = Dict{String, Any}("vendor" => String(h.vendor), "tool" => String(h.tool),
        "kernel" => h.kernel, "dispatches" => length(h), "counters" => copy(h.counters))
    duration = [d.duration_s for d in h.dispatches]
    _counter_stats!(out, "dispatch_s", duration)
    all(!ismissing, duration) && (out["dispatch_total_s"] = sum(duration))
    for field in (:device, :resources)
        dicts = [getfield(d, field) for d in h.dispatches]
        for k in sort!(collect(union((Set(keys(d)) for d in dicts)...)))
            occursin("__", k) && continue   # vendor-namespaced passthrough metadata stays on the dispatch
            v = get(first(dicts), k, missing)
            ismissing(v) && continue
            all(d -> isequal(get(d, k, missing), v), dicts) && (out[String(field) * "_" * k] = v)
        end
    end
    sv = [d.slots for d in h.dispatches]
    !ismissing(first(sv)) && all(isequal(first(sv)), sv) && (out["slots"] = first(sv))
    for c in h.counters
        _counter_stats!(out, "raw_" * c, h[c])
        u = h.units[c]
        ismissing(u) || (out["raw_" * c * "_unit"] = u)
    end
    for (k, v) in hw_counter_derived(h)
        _counter_stats!(out, "derived_" * k, v)
    end
    for (k, v) in h.provenance
        out["collection_" * k] = v
    end
    return out
end
