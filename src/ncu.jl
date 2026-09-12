const _NCU_COUNTERS = Dict(
    :issue => ["gpu__time_duration.sum", "smsp__inst_executed.sum", "smsp__thread_inst_executed.sum"],
    :occupancy => ["gpu__time_duration.sum", "sm__warps_active.avg.pct_of_peak_sustained_active"],
    :memory => ["gpu__time_duration.sum", "dram__bytes.sum"],
    :fp64 => ["gpu__time_duration.sum", "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum",
        "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum", "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum"],
    :l2 => ["gpu__time_duration.sum", "lts__t_sector_hit_rate.pct"],
)

"""
    COUNTER_SETS

Intent-based presets: `COUNTER_SETS[:amd][:issue]` / `COUNTER_SETS[:nvidia][:issue]`,
plus `:occupancy`, `:memory`, `:fp64`, and `:l2`. Each value is a [`CounterSet`](@ref).
AMD sets are validated on gfx942 with rocprofv3 1.1.0, ROCm 7.2.4, and each fits one pass
there. NVIDIA presets were collected on sm_120 with ncu 2025.4.1; their recorded pass counts
apply to a bounded FMA probe only, not to arbitrary workloads or other GPUs. Raw custom `metrics` remain available. Unsupported
metrics are errors from the collector, never silently replaced with different measurements.
"""
const COUNTER_SETS = Dict(
    :amd => Dict(k => CounterSet(copy(_AMD_COUNTERS[v]), ["gfx942 / rocprofv3 1.1.0 / ROCm 7.2.4"], 1,
        "One pass on the validated architecture; query rocprofv3 for other devices.") for (k, v) in
        (:issue => :sq_issue, :occupancy => :sq_waves, :memory => :l1_pipe, :fp64 => :fp64, :l2 => :l2)),
    :nvidia => Dict(k => CounterSet(copy(v), ["sm_120 / ncu 2025.4.1 / bounded FMA probe"], k in (:fp64, :l2) ? 3 : 1,
        "Observed passes for the validation probe only; query ncu on other devices and workloads.") for (k, v) in _NCU_COUNTERS),
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
            if c == "device__attribute_display_name"
                dispatches[i].device["name"] = value
                continue
            end
            setunit(c, u)
            v = try
                _counter_number(value)
            catch err
                err isa ArgumentError || rethrow()
                if startswith(c, "device__attribute_")
                    dispatches[i].device[c] = value
                    continue
                elseif startswith(c, "launch__")
                    dispatches[i].resources[c] = value
                    continue
                end
                rethrow()
            end
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
            d.resources[key] = get(r, c, missing)
        end
        blocks = get(r, "launch__grid_size", _ncu_dim_product(get(d.resources, "grid", missing)))
        threads = get(r, "launch__block_size", _ncu_dim_product(get(d.resources, "block", missing)))
        d.resources["grid_blocks"] = blocks
        d.resources["workgroup_size"] = threads
        d.resources["grid_size"] = blocks * threads  # common field: work-items, as on AMD
        dispatches[i] = HWDispatch(d.id, d.process_id, d.device_id, d.context_id, d.queue_id,
            d.kernel, d.start_s, duration, d.resources, d.device, d.slots)
    end
    return _build_counters(:nvidia, file, dispatches, rows, units; kernel, slots, provenance, required_metrics)
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
