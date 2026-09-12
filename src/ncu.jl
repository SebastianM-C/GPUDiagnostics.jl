"""
    NsightCompute()

Nsight Compute collector for an external NVIDIA workload. The CUDA extension selects
this automatically for `CUDABackend()`. Use it explicitly with [`hw_counter_command`](@ref)
or [`hw_counter_status`](@ref) when no CUDA runtime is loaded.
"""
struct NsightCompute <: HWCounterCollector end
_counter_vendor(::NsightCompute) = :nvidia
_counter_tool(::NsightCompute) = :ncu

function _counter_command_args(::NsightCompute, exe, metrics; dir, name, kernel,
        launch_skip::Integer = 0, launch_count::Union{Nothing, Integer} = nothing,
        clock_control::Symbol = :none, cache_control::Symbol = :all, replay_mode::Symbol = :kernel)
    launch_skip >= 0 || throw(ArgumentError("launch_skip must be non-negative"))
    launch_count === nothing || launch_count > 0 || throw(ArgumentError("launch_count must be positive"))
    clock_control in (:none, :base, :boost) || throw(ArgumentError("invalid clock_control"))
    cache_control in (:none, :all) || throw(ArgumentError("invalid cache_control"))
    replay_mode in (:kernel, :application) || throw(ArgumentError("replay_mode must be :kernel or :application"))
    args = String[exe, "--target-processes", "all", "--metrics", join(metrics, ','),
        "--clock-control", String(clock_control), "--cache-control", String(cache_control),
        "--replay-mode", String(replay_mode), "--launch-skip", string(launch_skip),
        "--csv", "--page", "raw", "--print-units", "base",
        "--log-file", joinpath(dir, name * "_ncu.csv"), "--export", joinpath(dir, name)]
    kernel === nothing || append!(args, ["--kernel-name", "regex:" * kernel])
    launch_count === nothing || append!(args, ["--launch-count", string(launch_count)])
    push!(args, "--")
    return args
end

const _NCU_COUNTERS = Dict(
    :issue => ["gpu__time_duration.sum", "smsp__inst_executed.sum", "smsp__thread_inst_executed.sum"],
    :occupancy => ["gpu__time_duration.sum", "sm__warps_active.avg.pct_of_peak_sustained_active"],
    :memory => ["gpu__time_duration.sum", "dram__bytes.sum"],
    :fp64 => ["gpu__time_duration.sum", "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum",
        "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum", "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum"],
    :l2 => ["gpu__time_duration.sum", "lts__t_sector_hit_rate.pct"],
)

# What each NVIDIA preset answers, and how its denominators differ from the AMD twin.
const _NCU_PRESET_NOTES = Dict{Symbol, String}(
    :issue => "Warp instructions (smsp__inst_executed.sum) and predicated-on THREAD instructions " *
        "(smsp__thread_inst_executed.sum) with the launch duration. Thread instructions / slots " *
        "(nvidia_insts_per_slot) is the per-slot dynamic count comparable to AMD's wave count × wave_size; " *
        "thread / (32 × warp) below 1 is divergence. Instruction counts are exact under replay; the " *
        "duration is ncu's serialized, cache-controlled replay, not a benchmark. One pass on sm_120.",
    :occupancy => "sm__warps_active.avg.pct_of_peak_sustained_active: achieved warps per SM as a " *
        "fraction of the SM maximum, averaged over ACTIVE cycles only (nvidia_active_occupancy). Idle " *
        "tails do not dilute it, so it is not the AMD elapsed-window estimate; hold it against " *
        "kernel_resources' theoretical occupancy, not against amd_elapsed_occupancy. One pass on sm_120.",
    :memory => "dram__bytes.sum: DRAM traffic of the launch; / duration is the achieved bandwidth. " *
        "There is no L1-pipe busy analogue in this preset — the nearest ncu metrics are the " *
        "l1tex__* family, passed as a custom `metrics` list. One pass on sm_120.",
    :fp64 => "Predicated-on DFMA / DADD / DMUL thread instructions: per-THREAD counts, so / slots " *
        "directly (no wave factor); fp64_flop_per_slot = 2·FMA + ADD + MUL. Add " *
        "sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active to `metrics` for " *
        "nvidia_fp64_pipe_peak_fraction, the question 'is the FP64 pipe the wall' (a kernel at 88 % " *
        "pipe fraction gained only 7–9 % from removing integer work). Three replay passes on sm_120.",
    :l2 => "lts__t_sector_hit_rate.pct: L2 hit rate over 32 B SECTOR lookups (nvidia_l2_sector_hit_rate). " *
        "Sectors are not AMD's TCC requests, so the two hit rates have different denominators and are " *
        "not directly comparable. Three replay passes on sm_120.",
)

"""
    COUNTER_SETS

Intent-based presets: `COUNTER_SETS[:amd][:issue]` / `COUNTER_SETS[:nvidia][:issue]`,
plus `:occupancy`, `:memory`, `:fp64`, and `:l2`. Each value is a [`CounterSet`](@ref) whose
`notes` say what the preset answers, how its raw values must be read (which counters are
per-wave, per-thread, event counts or quad-cycles) and the trap specific to it — read them
before interpreting a collection. AMD sets are validated on gfx942 with rocprofv3 1.1.0,
ROCm 7.2.4, and each fits one pass there; the L2 set is at the TCC block's four-counter
capacity, and exceeding a block's capacity aborts rocprofv3 and hangs the workload (hence
`timeout -k` in [`hw_counter_command`](@ref)). NVIDIA presets were collected on sm_120 with ncu
2025.4.1; their recorded pass counts apply to a bounded FMA probe only, not to arbitrary
workloads or other GPUs. Raw custom `metrics` remain available. Unsupported metrics are
errors from the collector, never silently replaced with different measurements.
"""
const COUNTER_SETS = Dict(
    :amd => Dict(k => CounterSet(copy(_AMD_COUNTERS[v]), ["gfx942 / rocprofv3 1.1.0 / ROCm 7.2.4"], 1,
        _AMD_PRESET_NOTES[k]) for (k, v) in
        (:issue => :sq_issue, :occupancy => :sq_waves, :memory => :l1_pipe, :fp64 => :fp64, :l2 => :l2)),
    :nvidia => Dict(k => CounterSet(copy(v), ["sm_120 / ncu 2025.4.1 / bounded FMA probe"], k in (:fp64, :l2) ? 3 : 1,
        _NCU_PRESET_NOTES[k]) for (k, v) in _NCU_COUNTERS),
)

const _NCU_IDENTITY = ("ID", "Process ID", "Process Name", "Host Name", "Kernel Name", "Context",
    "Stream", "Device", "Device Name", "CC", "Block Size", "Grid Size", "Section Name")

# ncu supports long metric/value exports and raw wide exports with a separate units row.
# Retain each format's native numeric values/units, including partial and N/A metrics.
function _ncu_records(file)
    header = String[]
    records = Vector{String}[]
    for line in eachline(file)
        isempty(strip(line)) && continue
        startswith(strip(line), "==") && begin
            occursin("ERROR", line) && throw(ArgumentError("ncu reported an error; the collection is incomplete"))
            continue
        end
        f = _csv_fields(rstrip(line, '\r'))
        if "ID" in f && "Kernel Name" in f
            isempty(header) || f == header || throw(ArgumentError("inconsistent ncu CSV headers"))
            header = f
            continue
        end
        isempty(header) && continue
        length(f) == length(header) || throw(ArgumentError("malformed ncu CSV row"))
        push!(records, f)
    end
    isempty(header) && throw(ArgumentError("no ncu CSV header found"))
    return header, records
end

function _ncu_seconds(value, unit)
    ismissing(value) && return missing
    scales = Dict("nsecond" => 1e-9, "ns" => 1e-9, "usecond" => 1e-6, "us" => 1e-6,
        "µs" => 1e-6, "msecond" => 1e-3, "ms" => 1e-3, "second" => 1.0, "s" => 1.0)
    ismissing(unit) && return missing
    haskey(scales, unit) || throw(ArgumentError("unsupported ncu duration unit $unit"))
    return value * scales[unit]
end

function _ncu_dim_product(value)
    ismissing(value) && return missing
    m = match(r"^\(\s*(\d+),\s*(\d+),\s*(\d+)\s*\)$", value)
    return m === nothing ? missing : prod(parse.(Int, m.captures))
end

function _parse_ncu(file; kernel = nothing, slots = nothing, device_overrides = Dict(), provenance = Dict(), required_metrics = nothing)
    header, records = _ncu_records(file)
    long = "Metric Name" in header
    long && foreach(c -> _csv_column(header, c, file), ("Metric Unit", "Metric Value"))
    units = Dict{String, Union{Missing, String}}()
    dispatches = HWDispatch[]
    rows = Dict{String, Union{Missing, Float64}}[]
    passes = Union{Missing, Int}[]
    seen = Dict{Tuple, Int}()
    function setunit(c, u)
        u = ismissing(u) || isempty(u) ? missing : String(u)
        haskey(units, c) && !isequal(units[c], u) && throw(ArgumentError("inconsistent units for $c"))
        units[c] = u
    end
    for row in records
        getv(n) = _csv_get(header, row, n)
        if !long && ismissing(getv("ID"))
            for (j, c) in enumerate(header)
                c in _NCU_IDENTITY || setunit(c, row[j])
            end
            continue
        end
        id = _counter_int(getv("ID"))
        ismissing(id) && throw(ArgumentError("missing ncu launch ID"))
        pid = _counter_int(getv("Process ID"))
        dev = _counter_int(getv("Device"))
        ctx = _counter_int(getv("Context"))
        stream = _counter_int(getv("Stream"))
        key = (pid, dev, ctx, stream, id)
        kname = getv("Kernel Name")
        ismissing(kname) && throw(ArgumentError("missing ncu kernel name"))
        if !haskey(seen, key)
            device = _device_override(Dict{String, Any}("name" => getv("Device Name"), "architecture" => getv("CC")), dev, device_overrides)
            resources = Dict{String, Any}("grid" => getv("Grid Size"), "block" => getv("Block Size"))
            push!(dispatches, HWDispatch(id, pid, dev, ctx, stream, kname, missing, missing, resources, device, missing))
            push!(rows, Dict{String, Union{Missing, Float64}}())
            push!(passes, missing)
            seen[key] = length(rows)
        end
        i = seen[key]
        dispatches[i].kernel == kname || throw(ArgumentError("conflicting kernel names for ncu dispatch $id"))
        for (key, col) in (("grid", "Grid Size"), ("block", "Block Size"))
            isequal(dispatches[i].resources[key], getv(col)) ||
                throw(ArgumentError("conflicting launch geometry for ncu dispatch $id"))
        end
        metrics = long ? [(String(getv("Metric Name")), getv("Metric Unit"), getv("Metric Value"))] :
            [(c, get(units, c, missing), row[j]) for (j, c) in enumerate(header) if c ∉ _NCU_IDENTITY]
        for (c, u, value) in metrics
            # `--page raw` exports everything ncu collected: every rollup of each requested metric,
            # and next to them ~170 device attributes, ~70 launch attributes, NVLink / C2C / NUMA
            # topology and the replay pass count. Those are not counters: they go to the dispatch's
            # `device` / `resources` (under their native names) and to `provenance["passes"]`, so
            # `counters` keeps only what the hardware measured.
            if c == "device__attribute_display_name"
                dispatches[i].device["name"] = value
                continue
            elseif any(pre -> startswith(c, pre), ("device__", "nvlink__", "c2clink__", "numa__"))
                dispatches[i].device[c] = _metadata_value(value)   # device attributes and topology
                continue
            elseif startswith(c, "launch__")
                dispatches[i].resources[c] = _metadata_value(value)
                continue
            elseif c == "profiler__replayer_passes"
                passes[i] = _counter_int(value)
                continue
            elseif startswith(c, "profiler__")
                continue
            end
            setunit(c, u)
            v = _counter_number(value)
            if haskey(rows[i], c)
                isequal(rows[i][c], v) || throw(ArgumentError("conflicting duplicate ncu metric $c for dispatch $id"))
            else
                rows[i][c] = v
            end
        end
    end
    for (i, d) in enumerate(dispatches)
        r = rows[i]
        duration = _ncu_seconds(get(r, "gpu__time_duration.sum", missing), get(units, "gpu__time_duration.sum", missing))
        !ismissing(duration) && duration < 0 && throw(ArgumentError("negative ncu duration"))
        for (key, c) in (("registers", "launch__registers_per_thread"), ("shared_mem_bytes", "launch__shared_mem_per_block_static"))
            d.resources[key] = get(d.resources, c, missing)
        end
        blocks = get(d.resources, "launch__grid_size", _ncu_dim_product(get(d.resources, "grid", missing)))
        threads = get(d.resources, "launch__block_size", _ncu_dim_product(get(d.resources, "block", missing)))
        d.resources["grid_blocks"] = blocks
        d.resources["workgroup_size"] = threads
        d.resources["grid_size"] = blocks * threads  # common field: work-items, as on AMD
        dispatches[i] = HWDispatch(d.id, d.process_id, d.device_id, d.context_id, d.queue_id,
            d.kernel, d.start_s, duration, d.resources, d.device, d.slots)
    end
    # The tool's own pass count is provenance when every dispatch agrees and the caller did not set it.
    prov = Dict{String, Any}(String(k) => v for (k, v) in pairs(provenance))
    if !haskey(prov, "passes") && !isempty(passes) && all(!ismissing, passes) && allequal(passes)
        prov["passes"] = first(passes)
    end
    return _build_counters(NsightCompute(), file, dispatches, rows, units; kernel, slots, provenance = prov, required_metrics)
end

# A metadata cell: numeric where it parses as a number, the string otherwise, missing when empty.
function _metadata_value(value)
    ismissing(value) && return missing
    v = tryparse(Float64, replace(strip(String(value)), "," => ""))
    return v === nothing ? String(value) : isinteger(v) ? Int(v) : v
end

function _nvidia_derived(raw, d, units)
    out = Dict{String, Union{Missing, Float64}}()
    function percent(key, c)
        haskey(raw, c) || return
        u = get(units, c, missing)
        ismissing(u) || u in ("%", "percent") || throw(ArgumentError("expected percent units for $c"))
        out[key] = raw[c] / 100
    end
    # Active-cycle SM averaging and sector lookups are not the AMD elapsed-device
    # window and TCC request denominator. Keep these metrics explicitly qualified.
    percent("nvidia_active_occupancy", "sm__warps_active.avg.pct_of_peak_sustained_active")
    percent("nvidia_l2_sector_hit_rate", "lts__t_sector_hit_rate.pct")
    percent("nvidia_fp64_pipe_peak_fraction", "sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active")
    if haskey(raw, "sm__cycles_elapsed.avg.per_second")
        u = get(units, "sm__cycles_elapsed.avg.per_second", missing)
        scale = get(Dict("cycle/second" => 1e-9, "cycle/nsecond" => 1.0, "hz" => 1e-9, "GHz" => 1.0), u, missing)
        out["nvidia_elapsed_clock_GHz"] = raw["sm__cycles_elapsed.avg.per_second"] * scale
    end
    if !ismissing(d.slots)
        function instructions(c)
            u = get(units, c, missing)
            # Imported CSV can use scaled units even though our collector requests base units.
            scale = get(Dict("inst" => 1.0, "Kinst" => 1e3, "Minst" => 1e6, "Ginst" => 1e9), u, missing)
            return raw[c] * scale
        end
        for (op, native) in (("fma", "dfma"), ("add", "dadd"), ("mul", "dmul"))
            c = "smsp__sass_thread_inst_executed_op_" * native * "_pred_on.sum"
            haskey(raw, c) && (out["insts_per_slot_fp64_" * op] = _safe_ratio(instructions(c), d.slots))
        end
        ks = ["insts_per_slot_fp64_" * op for op in ("fma", "add", "mul")]
        all(k -> haskey(out, k), ks) && (out["fp64_flop_per_slot"] = 2out[ks[1]] + out[ks[2]] + out[ks[3]])
        haskey(raw, "smsp__thread_inst_executed.sum") &&
            (out["nvidia_insts_per_slot"] = _safe_ratio(instructions("smsp__thread_inst_executed.sum"), d.slots))
    end
    return out
end
