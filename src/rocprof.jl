const _AMD_COUNTERS = Dict{Symbol, Vector{String}}(
    :sq_issue => ["GRBM_GUI_ACTIVE", "SQ_INSTS_VMEM_RD", "SQ_INSTS_VMEM_WR", "SQ_INSTS_SMEM",
        "SQ_INSTS_SALU", "SQ_INSTS_BRANCH", "SQ_INSTS_VALU_INT64", "SQ_INSTS_VALU"],
    :sq_waves => ["SQ_WAVES", "SQ_BUSY_CYCLES", "GRBM_GUI_ACTIVE", "SQ_WAVE_CYCLES",
        "SQ_ACTIVE_INST_ANY", "SQ_ACTIVE_INST_VALU", "SQ_WAIT_INST_ANY", "SQ_WAIT_ANY"],
    :l1_pipe => ["GRBM_GUI_ACTIVE", "TA_TA_BUSY_sum", "TD_TD_BUSY_sum", "TCP_PENDING_STALL_CYCLES_sum",
        "TCP_TOTAL_READ_sum", "TCP_TCC_READ_REQ_sum", "TCP_TOTAL_CACHE_ACCESSES_sum"],
    :fp64 => ["GRBM_GUI_ACTIVE", "SQ_INSTS_VALU_FMA_F64", "SQ_INSTS_VALU_ADD_F64",
        "SQ_INSTS_VALU_MUL_F64", "SQ_INSTS_VALU_TRANS_F64"],
    :l2 => ["GRBM_GUI_ACTIVE", "TCC_HIT_sum", "TCC_MISS_sum", "TCC_EA0_RDREQ_sum", "TCC_EA0_RDREQ_DRAM_sum"],
)

function _agent_id(s)
    ismissing(s) && return missing
    m = match(r"^(?:Agent\s+)?(\d+)$", strip(s))
    m === nothing && throw(ArgumentError("invalid agent id $s"))
    return parse(Int, m[1])
end

function _rocprof_devices(path)
    devices = Dict{Int, Dict{String, Any}}()
    isfile(path) || return devices
    header, rows = _read_csv(path)
    for row in rows
        isequal(_csv_get(header, row, "Agent_Type"), "GPU") || continue
        id = _counter_int(_csv_get(header, row, "Node_Id"))
        ismissing(id) && continue
        d = Dict{String, Any}("architecture" => _csv_get(header, row, "Name"),
            "name" => _csv_get(header, row, "Product_Name"))
        for (key, column) in (("n_cu", "Cu_Count"), ("n_xcd", "Num_Xcc"),
                ("n_se", "Num_Shader_Banks"), ("n_simd", "Simd_Count"),
                ("wave_size", "Wave_Front_Size"), ("max_waves_per_cu", "Max_Waves_Per_Cu"))
            n = _counter_int(_csv_get(header, row, column))
            d[key] = ismissing(n) || n <= 0 ? missing : n
        end
        devices[id] = d
    end
    return devices
end

function _parse_rocprof(file; kernel = nothing, slots = nothing,
        device_overrides = Dict(), provenance = Dict(), required_metrics = nothing)
    header, records = _read_csv(file)
    for c in ("Dispatch_Id", "Kernel_Name", "Counter_Name", "Counter_Value")
        _csv_column(header, c, file)
    end
    agents = _rocprof_devices(joinpath(dirname(file), _collection_name(file, :amd) * "_agent_info.csv"))
    dispatches = HWDispatch[]
    rows = Dict{String, Union{Missing, Float64}}[]
    seen = Dict{Tuple, Int}()
    for row in records
        getv(n) = _csv_get(header, row, n)
        id = _counter_int(getv("Dispatch_Id"))
        ismissing(id) && throw(ArgumentError("missing Dispatch_Id"))
        pid = _counter_int(getv("Process_Id"))
        dev = _agent_id(getv("Agent_Id"))
        queue = _counter_int(getv("Queue_Id"))
        key = (pid, dev, queue, id)
        kname = getv("Kernel_Name")
        ismissing(kname) && throw(ArgumentError("missing kernel name"))
        start = _counter_int(getv("Start_Timestamp"))
        stop = _counter_int(getv("End_Timestamp"))
        duration = ismissing(start) || ismissing(stop) ? missing : (stop - start) / 1e9
        !ismissing(duration) && duration < 0 && throw(ArgumentError("negative dispatch duration"))
        resources = Dict{String, Any}()
        for (k, c) in (("grid_size", "Grid_Size"), ("workgroup_size", "Workgroup_Size"),
                ("vgpr_count", "VGPR_Count"), ("agpr_count", "Accum_VGPR_Count"),
                ("sgpr_count", "SGPR_Count"), ("lds_bytes", "LDS_Block_Size"), ("scratch_bytes", "Scratch_Size"))
            resources[k] = _reported(_counter_int(getv(c)))
        end
        if !haskey(seen, key)
            device = _device_override(ismissing(dev) ? Dict{String, Any}() : get(agents, dev, Dict{String, Any}()), dev, device_overrides)
            push!(dispatches, HWDispatch(id, pid, dev, missing, queue, kname,
                ismissing(start) ? missing : start / 1e9, duration, resources, device, missing))
            push!(rows, Dict{String, Union{Missing, Float64}}())
            seen[key] = length(rows)
        end
        i = seen[key]
        d = dispatches[i]
        d.kernel == kname && isequal(d.duration_s, duration) && isequal(d.resources, resources) ||
            throw(ArgumentError("conflicting metadata for dispatch $id; collections cannot be merged by dispatch id"))
        counter = getv("Counter_Name")
        ismissing(counter) && throw(ArgumentError("missing counter name"))
        haskey(rows[i], counter) && throw(ArgumentError("duplicate counter $counter for dispatch $id"))
        rows[i][counter] = _counter_number(getv("Counter_Value"))
    end
    # Preserve tool file order, including distinct processes/devices with the same dispatch ID.
    return _build_counters(:amd, file, dispatches, rows, Dict(); kernel, slots, provenance, required_metrics)
end

# gfx942 normalization is fixture-validated. Do not silently apply quad-cycle or
# die-sum assumptions to other architectures. Raw data and dimension-free ratios
# remain usable everywhere; metadata can explicitly identify a known architecture.
function _amd_derived(raw, d)
    out = Dict{String, Union{Missing, Float64}}()
    getv(c) = get(raw, c, missing)
    prop(c) = get(d.device, c, missing)
    ratio(key, a, b) = haskey(raw, a) && haskey(raw, b) && (out[key] = _safe_ratio(raw[a], raw[b]))
    if !ismissing(d.slots)
        for c in keys(raw)
            startswith(c, "SQ_INSTS_") || continue
            out["amd_insts_per_slot_" * lowercase(c[10:end])] = _safe_ratio(raw[c] * prop("wave_size"), d.slots)
        end
        for (op, native) in (("fma", "FMA"), ("add", "ADD"), ("mul", "MUL"))
            c = "SQ_INSTS_VALU_" * native * "_F64"
            haskey(raw, c) && (out["insts_per_slot_fp64_" * op] = _safe_ratio(raw[c] * prop("wave_size"), d.slots))
        end
        fma, add, mul = "SQ_INSTS_VALU_FMA_F64", "SQ_INSTS_VALU_ADD_F64", "SQ_INSTS_VALU_MUL_F64"
        if all(c -> haskey(raw, c), (fma, add, mul))
            out["fp64_flop_per_slot"] = _safe_ratio((2raw[fma] + raw[add] + raw[mul]) * prop("wave_size"), d.slots)
        end
    end
    arch = prop("architecture")
    known = !ismissing(arch) && first(split(arch, ':')) == "gfx942"
    cycles = known ? _safe_ratio(getv("GRBM_GUI_ACTIVE"), prop("n_xcd")) : missing
    if haskey(raw, "GRBM_GUI_ACTIVE")
        out["amd_active_clock_GHz"] = _safe_ratio(cycles, d.duration_s) / 1e9
        for c in keys(raw)
            m = match(r"^([A-Z]+)_[A-Z]+_BUSY_sum$", c)
            m === nothing && continue
            out["amd_" * lowercase(m[1]) * "_busy"] = _safe_ratio(raw[c], cycles * prop("n_cu"))
        end
        haskey(raw, "TCP_PENDING_STALL_CYCLES_sum") &&
            (out["amd_tcp_pending_stall"] = _safe_ratio(raw["TCP_PENDING_STALL_CYCLES_sum"], cycles * prop("n_cu")))
        if haskey(raw, "SQ_WAVE_CYCLES")
            resident = _safe_ratio(4raw["SQ_WAVE_CYCLES"], cycles)
            out["amd_resident_waves"] = resident
            out["amd_waves_per_cu"] = _safe_ratio(resident, prop("n_cu"))
            # This is elapsed-active-device-window occupancy, not NVIDIA's per-SM active-cycle average.
            out["amd_elapsed_occupancy"] = _safe_ratio(resident, prop("n_cu") * prop("max_waves_per_cu"))
        end
        haskey(raw, "SQ_BUSY_CYCLES") && (out["amd_sq_busy"] = _safe_ratio(raw["SQ_BUSY_CYCLES"], cycles * prop("n_se")))
    end
    ratio("amd_l1_miss", "TCP_TCC_READ_REQ_sum", "TCP_TOTAL_CACHE_ACCESSES_sum")
    ratio("amd_l2_dram_read_frac", "TCC_EA0_RDREQ_DRAM_sum", "TCC_EA0_RDREQ_sum")
    haskey(raw, "TCC_HIT_sum") && haskey(raw, "TCC_MISS_sum") &&
        (out["amd_l2_request_hit_rate"] = _safe_ratio(raw["TCC_HIT_sum"], raw["TCC_HIT_sum"] + raw["TCC_MISS_sum"]))
    for (key, c) in (("wave_wait_frac", "SQ_WAIT_ANY"), ("wave_wait_inst_frac", "SQ_WAIT_INST_ANY"),
            ("wave_active_inst_frac", "SQ_ACTIVE_INST_ANY"), ("wave_active_valu_frac", "SQ_ACTIVE_INST_VALU"))
        ratio("amd_" * key, c, "SQ_WAVE_CYCLES")
    end
    haskey(raw, "SQ_WAVE_CYCLES") && haskey(raw, "SQ_WAVES") &&
        (out["amd_wave_cycles"] = known ? _safe_ratio(4raw["SQ_WAVE_CYCLES"], raw["SQ_WAVES"]) : missing)
    return out
end
