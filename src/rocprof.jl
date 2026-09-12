# AMD per-dispatch hardware counters through rocprofv3 (ROCm 6.2+ / 7.x). AMDGPU.jl has no
# in-process counter API (nothing like NVIDIA's GPM), so the model is the profiler's: rocprofv3
# wraps a whole process (or, in recent releases, attaches to one) and writes
# `<name>_counter_collection.csv` (one row per dispatch × counter, with the dispatch's timestamps
# and register / LDS / scratch footprint), `<name>_kernel_trace.csv` and `<name>_agent_info.csv`
# (the device: CU / XCD / SE / SIMD counts, wavefront size). Kernels run at speed under `--pmc`;
# the overhead is the counter read per dispatch, so profiled durations sit close to device-event
# timings (26.8–27.4 ms vs a 28.0 ms event median on gfx942) — unlike ncu's replayed durations.
#
# How the raw values must be read (the normalisation rocprofv3's OWN derived metrics use on
# gfx942, `rocprofv3 --list-avail`, rocprofv3 1.1.0):
#
#     OccupancyPercent = 400·Σ SQ_WAVE_CYCLES / max_xcc(GRBM_GUI_ACTIVE) / CU_NUM / 32
#     SALUBusy         = 100·Σ SQ_INST_CYCLES_SALU / CU_NUM / max_xcc(GRBM_GUI_ACTIVE)
#
# (1) Every counter is reported SUMMED over its hardware dimensions. `GRBM_GUI_ACTIVE` has
#     DIMENSION_XCC[0:7] on the MI300X, so the CSV value is 8× the cycles one die saw: a 40 ms
#     dispatch shows 5.7e8 "cycles" = 14 GHz until divided by the dies. The dispatch's cycle count
#     is `GRBM_GUI_ACTIVE / n_xcd` (`Num_Xcc` in the agent info; 1 on single-die parts), and every
#     per-cycle rate (unit busy, resident waves, clock) uses it. 14 GHz is the sanity check that
#     the division was forgotten; "/38" and "/(8 × 304)" are the same per-CU normalisation.
# (2) `*_sum` unit counters are summed over every instance on the device (one TA / TD / TCP per
#     CU), so a unit's busy fraction is `X_BUSY_sum / (cycles × n_cu)`. They are EVENT counts, not
#     per-wave issues: never multiply them by the wavefront size (doing so once inflated per-slot
#     L1 accesses 64×).
# (3) The SQ wave-cycle counters (`SQ_WAVE_CYCLES`, `SQ_WAIT_ANY`, `SQ_WAIT_INST_ANY`,
#     `SQ_ACTIVE_INST_*`) are in QUAD-cycles (4 clocks) on gfx9: ratios between them are unit-free,
#     resident waves = 4·SQ_WAVE_CYCLES / cycles. `SQ_BUSY_CYCLES` is plain cycles per shader engine.
# (4) `SQ_INSTS_*` count per-WAVE instruction issues; × wavefront size (64 on CDNA) / slots is the
#     per-slot count a static instruction mix can be held against. Validated on gfx942: the FP64
#     classes per slot (219.1 / 44.0 / 178.1 / 15.0) matched the static hot loop exactly.
# (5) `cycles / duration` is the MEAN ACTIVE clock, not the engine clock: idle dies count no cycles,
#     so it is a lower bound and meaningless for sub-ms dispatches. The MI300X is power-managed
#     (1.70 GHz at 7 waves/CU, 1.25 GHz at 14 waves/CU on the same kernel, against the 750 W board
#     cap; sysfs agreed to 0.01 GHz), so compare two runs of the SAME work in CYCLES, not seconds.
# (6) Counter capacity is per hardware block and per pass. The TCC block takes FOUR counters on
#     gfx942; a fifth makes rocprofv3 log "Request exceeds the capabilities of the hardware to
#     collect", abort with SIGABRT and leave the profiled child hung — `hw_counter_command` wraps
#     the collection in `timeout -k` for exactly that failure (the child then exits 137).
# The parser needs neither AMDGPU.jl nor a GPU; the gfx942 fixtures under test/fixtures/rocprof
# are trimmed real MI300X collections.

"""
    RocprofV3()

rocprofv3 collector for an external AMD workload. The AMDGPU extension selects this
automatically for `ROCBackend()`. Use it explicitly with [`hw_counter_command`](@ref)
or [`hw_counter_status`](@ref) when no AMDGPU runtime is loaded.
"""
struct RocprofV3 <: HWCounterCollector end
_counter_vendor(::RocprofV3) = :amd
_counter_tool(::RocprofV3) = :rocprofv3

function _counter_command_args(::RocprofV3, exe, metrics; dir, name, kernel, kernel_trace::Bool = true)
    args = String[exe]
    kernel_trace && push!(args, "--kernel-trace")
    kernel === nothing || append!(args, ["--kernel-include-regex", kernel])
    append!(args, ["--pmc"; metrics; "--output-format"; "csv"; "-d"; dir; "-o"; name; "--"])
    return args
end

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

# What each AMD preset answers, and the trap specific to it. Surfaced through `CounterSet.notes`.
const _AMD_PRESET_NOTES = Dict{Symbol, String}(
    :issue => "Instruction issue per wave by class (SQ_INSTS_VMEM_RD/_WR, _SMEM, _SALU, _BRANCH, " *
        "_VALU_INT64, _VALU) plus GRBM_GUI_ACTIVE. SQ_INSTS_* are per-WAVE issues: × wave_size / slots " *
        "(`amd_insts_per_slot_<class>`) is the dynamic per-slot mix to hold against the static " *
        "kernel_instruction_mix of the hot loop — is the kernel load-, integer- or branch-heavy. " *
        "One pass on gfx942 (rocprofv3 1.1.0, ROCm 7.2.4).",
    :occupancy => "Wave residency and what waves do: SQ_WAVES, SQ_BUSY_CYCLES, SQ_WAVE_CYCLES, " *
        "SQ_ACTIVE_INST_ANY/_VALU, SQ_WAIT_INST_ANY, SQ_WAIT_ANY plus GRBM_GUI_ACTIVE. The wave-cycle " *
        "counters are QUAD-cycles on gfx9 (resident waves = 4·SQ_WAVE_CYCLES / cycles; ratios between " *
        "them are unit-free). amd_wave_wait_frac − amd_wave_wait_inst_frac is the dependency / " *
        "memory-latency wait: waves mostly waiting on data (latency-bound) vs on the arbiter " *
        "(issue-bound). amd_elapsed_occupancy is rocprofv3's OccupancyPercent / 100, an elapsed-window " *
        "figure to hold against kernel_resources' theoretical occupancy. One pass on gfx942.",
    :memory => "The vector-memory pipe: TA_TA_BUSY_sum, TD_TD_BUSY_sum, TCP_PENDING_STALL_CYCLES_sum, " *
        "TCP_TOTAL_READ_sum, TCP_TCC_READ_REQ_sum, TCP_TOTAL_CACHE_ACCESSES_sum plus GRBM_GUI_ACTIVE. " *
        "*_sum counters are summed over every instance (one TA/TD/TCP per CU): busy = X_sum / " *
        "(cycles × n_cu). They are EVENT counts, not per-wave issues — do not multiply by wave_size. " *
        "amd_l1_miss = L1→L2 read requests per L1 access: does the working set fit L1; an in-order " *
        "pipe at 85–95 % TD busy with waves 70 % waiting is memory-pipe-bound, not FLOP-bound. One pass on gfx942.",
    :fp64 => "FP64 VALU issue per wave: SQ_INSTS_VALU_FMA_F64, _ADD_F64, _MUL_F64, _TRANS_F64 plus " *
        "GRBM_GUI_ACTIVE. × wave_size / slots is the hardware's own FLOP count per slot " *
        "(fp64_flop_per_slot = 2·FMA + ADD + MUL; TRANS is reported separately as " *
        "amd_insts_per_slot_valu_trans_f64). Validated on gfx942: 219.1 / 44.0 / 178.1 / 15.0 per slot " *
        "matched the static hot loop + second Newton pass exactly, bit-identical across dispatches; a " *
        "`c = a + b·1.5` probe gave 1.00 ADD + 1.00 MUL per element. gfx1100 exposes no F64 counters. One pass on gfx942.",
    :l2 => "The L2 (TCC) and what leaves the die: TCC_HIT_sum, TCC_MISS_sum, TCC_EA0_RDREQ_sum (L2→fabric " *
        "reads), TCC_EA0_RDREQ_DRAM_sum (of those, to HBM) plus GRBM_GUI_ACTIVE: does the working set live " *
        "in L2 or in HBM (amd_l2_request_hit_rate, amd_l2_dram_read_frac; the medians / slots are requests " *
        "per slot). The TCC block collects FOUR counters per pass on gfx942: a fifth (TCC_REQ_sum, " *
        "TCC_READ_sum) makes rocprofv3 log \"Request exceeds the capabilities of the hardware to collect\", " *
        "abort with SIGABRT and leave the profiled child hung until `timeout -k` kills it (exit 137). One pass on gfx942.",
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
    return _build_counters(RocprofV3(), file, dispatches, rows, Dict(); kernel, slots, provenance, required_metrics)
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
