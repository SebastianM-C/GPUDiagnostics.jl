# ── Report layer: flat dicts for manifests, readable summaries, Tables.jl ──────────────────
#
# Every consumer of this package has to reduce its results into something storable and
# something readable. `diagnostics_dict` is the storable form: a flat `Dict{String, Any}` of
# TOML-safe scalars (Int / Float64 / Bool / String, or vectors of those) under stable keys — a
# key is only added within a schema version, and a value the backend could not report (`missing`) is
# OMITTED rather than written as a sentinel, so manifests written months apart merge and
# compare by key. `show(io, MIME"text/plain"(), x)` is the readable form of the same content.

"""
    GPUDIAGNOSTICS_SCHEMA

Version of the key layout [`diagnostics_dict`](@ref) writes. Every dict carries it under
`"gpudiagnostics_schema"` (unprefixed, so a manifest merging several dicts holds it once). Keys
are additive within a schema; version 2 introduces the unified hardware-counter layout.
"""
const GPUDIAGNOSTICS_SCHEMA = 2

"""
    diagnostics_dict(x; prefix = "") -> Dict{String, Any}

Flat, TOML-safe dictionary of a result — `KernelResources`, `KernelInstructionMix` /
`InstructionMix`, `IRMix`, `FP64IssueFloor`, `LaunchTimer`, `GPUTelemetry`, `HWCounters` —
with every key prefixed by `prefix` (the convention used by the first consumer: `kernel_`,
`kernel_mix_`, `kernel_ir_`, `sampler_`, `hw_`), ready to `merge!` into a run manifest.
Values are `Int`, `Float64`, `Bool`, `String` or vectors of those; `Symbol`s become strings;
a `missing` value is omitted (never written as a sentinel), as are `nothing` fields. Every
dict also carries `"gpudiagnostics_schema" => `[`GPUDIAGNOSTICS_SCHEMA`](@ref).

The instruction-mix dict does not include the IR counts: call
`diagnostics_dict(m.ir; prefix = "kernel_ir_")` for those, so the two key families stay
separately prefixed.
"""
function diagnostics_dict end

_dictval(x::Symbol) = String(x)
_dictval(x::Union{Integer, AbstractFloat, Bool, AbstractString}) = x
_dictval(x::AbstractVector) = [_dictval(v) for v in x]
_dictval(x) = string(x)

function _put!(d::Dict{String, Any}, prefix::AbstractString, k, v)
    (v === missing || v === nothing) && return d
    d[prefix * String(k)] = _dictval(v)
    return d
end
_newdict() = Dict{String, Any}("gpudiagnostics_schema" => GPUDIAGNOSTICS_SCHEMA)

# ---- KernelResources
function diagnostics_dict(r::KernelResources; prefix::AbstractString = "")
    d = _newdict()
    for k in (:name, :block_size, :registers, :local_mem_bytes, :shared_mem_bytes, :const_mem_bytes, :max_threads_per_block)
        _put!(d, prefix, k, getfield(r, k))
    end
    if !ismissing(r.occupancy)
        o = r.occupancy
        for k in (:active_blocks_per_sm, :active_warps_per_sm, :max_warps_per_sm, :warp_size, :max_threads_per_sm, :shared_mem_per_sm)
            _put!(d, prefix, k, getfield(o, k))
        end
        _put!(d, prefix, :occupancy, o.fraction)
    end
    for (k, v) in r.isa
        _put!(d, prefix, "isa_" * k, v)
    end
    return d
end

# ---- instruction mix
function diagnostics_dict(m::InstructionMix; prefix::AbstractString = "")
    d = _newdict()
    for k in (:vendor, :total, :fp64, :coverage, :unclassified, :blocks, :hot_loop_confidence, :llvm_loops_agree)
        _put!(d, prefix, k, getfield(m, k))
    end
    _put!(d, prefix, :loops, length(m.loops))
    for c in MIX_CLASSES
        _put!(d, prefix, c, m.counts[c])
    end
    if m.hot_loop !== nothing
        h = m.hot_loop
        _put!(d, prefix, :hot_loop_total, h.total)
        _put!(d, prefix, :hot_loop_fp64, _fp64_total(h.counts))
        _put!(d, prefix, :hot_loop_depth, h.depth)
        for c in MIX_CLASSES
            _put!(d, prefix, "hot_loop_" * String(c), h.counts[c])
        end
    end
    return d
end
function diagnostics_dict(m::KernelInstructionMix; prefix::AbstractString = "")
    d = diagnostics_dict(m.mix; prefix)
    for k in (:name, :target, :native, :registers)
        _put!(d, prefix, k, getfield(m, k))
    end
    return d
end
function diagnostics_dict(m::IRMix; prefix::AbstractString = "")
    d = _newdict()
    for k in (:target, :total, :fp64)
        _put!(d, prefix, k, getfield(m, k))
    end
    for c in IR_CLASSES
        _put!(d, prefix, c, m.counts[c])
    end
    return d
end
function diagnostics_dict(f::FP64IssueFloor; prefix::AbstractString = "")
    d = _newdict()
    for k in fieldnames(FP64IssueFloor)
        _put!(d, prefix, k, getfield(f, k))
    end
    return d
end

# ---- LaunchTimer: one vector per statistic, in ascending device order (the layout the first
# consumer's manifests already use).
function diagnostics_dict(t::LaunchTimer; prefix::AbstractString = "")
    d = _newdict()
    lt = launch_times(t)
    devs = sort!(collect(keys(lt)))
    isempty(devs) && return d
    per = [lt[dv] for dv in devs]
    _put!(d, prefix, :devices, devs)
    _put!(d, prefix, :launches, map(length, per))
    _put!(d, prefix, :s, map(sum, per))
    _put!(d, prefix, :first_s, map(first, per))
    _put!(d, prefix, :median_s, map(v -> Float64(Statistics.median(v)), per))
    _put!(d, prefix, :max_s, map(maximum, per))
    return d
end

# ---- GPUTelemetry: the stats plus the sampling provenance
function diagnostics_dict(t::GPUTelemetry; prefix::AbstractString = "", kwargs...)
    d = _newdict()
    for k in (:ticks, :dt, :window, :first_sample_s, :starved, :counters)
        _put!(d, prefix, k, getfield(t, k))
    end
    for (k, v) in gpu_telemetry_stats(t; kwargs...)
        _put!(d, prefix, k, v)
    end
    return d
end

# ---- HWCounters: the summary, prefixed
function diagnostics_dict(rc::HWCounters; prefix::AbstractString = "")
    d = _newdict()
    for (k, v) in hw_counter_summary(rc)
        _put!(d, prefix, k, v)
    end
    return d
end

# ── Readable summaries ──────────────────────────────────────────────────────────────────────

_fmt3(x) = ismissing(x) ? "—" : string(round(x; digits = 3))

function _show_counts(io, counts, classes; indent = "  ")
    nz = [c for c in classes if counts[c] != 0]
    isempty(nz) && return println(io, indent, "(no instructions)")
    w = maximum(length(String(c)) for c in nz)
    for c in nz
        println(io, indent, rpad(String(c), w + 2), counts[c])
    end
    return nothing
end

Base.show(io::IO, m::InstructionMix) = print(io, "InstructionMix(", m.vendor, ": ", m.total, " instructions, fp64 ", m.fp64,
    ", coverage ", _fmt3(m.coverage), ", ", length(m.loops), " loops",
    m.hot_loop === nothing ? "" : ", hot loop $(m.hot_loop.total) ($(m.hot_loop_confidence))", ")")
function Base.show(io::IO, ::MIME"text/plain", m::InstructionMix)
    println(io, "InstructionMix (", m.vendor, "): ", m.total, " instructions in ", m.blocks, " blocks, coverage ",
        _fmt3(m.coverage), ", ", length(m.loops), " loop", length(m.loops) == 1 ? "" : "s",
        m.hot_loop === nothing ? ", no hot loop" : ", hot loop $(m.hot_loop.header) (confidence $(m.hot_loop_confidence))")
    # class × column table: whole kernel, then the hot loop (all / own blocks), then the other loops
    cols = Pair{String, MixCounts}["total (static)" => m.counts]
    if m.hot_loop !== nothing
        push!(cols, "hot loop" => m.hot_loop.counts, "hot loop excl." => m.hot_loop.exclusive_counts)
    end
    for l in m.loops
        (m.hot_loop !== nothing && l.header == m.hot_loop.header) && continue
        push!(cols, "$(l.header) d$(l.depth)" => l.counts)
    end
    w(nm) = max(16, length(nm) + 2)
    println(io, rpad("class", 14), join(lpad(first(c), w(first(c))) for c in cols))
    for c in MIX_CLASSES
        any(cnt[c] != 0 for (_, cnt) in cols) || continue
        println(io, rpad(String(c), 14), join(lpad(string(cnt[c]), w(nm)) for (nm, cnt) in cols))
    end
    println(io, rpad("fp64 (all)", 14), join(lpad(string(_fp64_total(cnt)), w(nm)) for (nm, cnt) in cols))
    println(io, rpad("TOTAL", 14), join(lpad(string(sum(cnt)), w(nm)) for (nm, cnt) in cols))
    if !isempty(m.loops)
        println(io, "loop nest (largest first; total = one pass, nested loops counted once; excl = own blocks):")
        for l in m.loops
            println(io, "  ", rpad(l.header, 12), " depth ", l.depth, "  blocks ", lpad(l.blocks, 4), "  total ", lpad(l.total, 6),
                "  excl ", lpad(l.exclusive_total, 6), "  fp64 ", lpad(_fp64_total(l.counts), 5), "  waits ", lpad(l.counts.wait, 4),
                "  loads ", lpad(l.counts.mem_load, 4))
        end
    end
    other = sort!([(k, v) for (k, v) in m.opcodes if _classify(k, m.vendor) in (:other, :unclassified)]; by = x -> -x[2])
    isempty(other) || print(io, "'other': ", join(("$k×$v" for (k, v) in other[1:min(end, 10)]), ", "))
    isempty(m.unclassified_opcodes) || print(io, "\nunclassified: ", join(("$k×$v" for (k, v) in sort!(collect(m.unclassified_opcodes); by = x -> -x[2])), ", "))
    return nothing
end
Base.show(io::IO, m::KernelInstructionMix) = print(io, "KernelInstructionMix(", m.name, " for ", m.target,
    m.native ? "" : " (cross-compiled)", ": ", m.total, " instructions, fp64 ", m.fp64,
    m.hot_loop === nothing ? "" : ", hot loop $(m.hot_loop.total)", ")")
function Base.show(io::IO, mime::MIME"text/plain", m::KernelInstructionMix)
    println(io, "KernelInstructionMix: ", m.name, " for ", m.target, m.native ? " (this device)" : " (cross-compiled)",
        ", registers ", ismissing(m.registers) ? "— (not in the listing)" : string(m.registers))
    show(io, mime, m.mix)
    if m.ir !== nothing
        println(io)
        show(io, mime, m.ir)
    end
    return nothing
end
Base.show(io::IO, m::IRMix) = print(io, "IRMix(", m.target, ": ", m.total, " IR ops, fp64 ", m.fp64, ", ", length(m.functions), " functions)")
function Base.show(io::IO, ::MIME"text/plain", m::IRMix)
    println(io, "IRMix (optimized LLVM IR for ", m.target, "): ", m.total, " ops in ", length(m.functions), " functions, fp64 ", m.fp64)
    _show_counts(io, m.counts, IR_CLASSES)
    return nothing
end
function Base.show(io::IO, f::FP64IssueFloor)
    print(io, "FP64IssueFloor(", f.scope, ": ", f.fp64_per_slot, " FP64/slot × ", f.n_slots, " slots at ",
        round(f.peak_fp64_flops / 1e12; digits = 2), " TFLOP/s ⇒ ", round(f.floor_s * 1e3; digits = 3), " ms",
        ismissing(f.fp64_issue_fraction) ? "" : " = $(round(100 * f.fp64_issue_fraction; digits = 1)) % of the launch", ")")
end
function Base.show(io::IO, ::MIME"text/plain", f::FP64IssueFloor)
    show(io, f)
    print(io, "\n  confidence ", f.confidence, "\n  assumptions: ", f.assumptions)
    return nothing
end
function Base.show(io::IO, t::LaunchTimer)
    n = sum(length, values(t.lanes); init = 0)
    print(io, "LaunchTimer(", n, " launches on ", length(t.lanes), " device", length(t.lanes) == 1 ? "" : "s", ")")
end
function Base.show(io::IO, ::MIME"text/plain", t::LaunchTimer)
    lt = launch_times(t)
    if isempty(lt)
        print(io, "LaunchTimer: no launches recorded")
        return nothing
    end
    println(io, "LaunchTimer: per-launch device time")
    println(io, "  ", rpad("device", 8), rpad("launches", 10), rpad("first_s", 12), rpad("median_s", 12), rpad("max_s", 12), "total_s")
    for dv in sort!(collect(keys(lt)))
        v = lt[dv]
        println(io, "  ", rpad(dv, 8), rpad(length(v), 10), rpad(round(first(v); sigdigits = 4), 12),
            rpad(round(Statistics.median(v); sigdigits = 4), 12), rpad(round(maximum(v); sigdigits = 4), 12), round(sum(v); sigdigits = 4))
    end
    return nothing
end
function Base.show(io::IO, ::MIME"text/plain", t::GPUTelemetry)
    println(io, "GPUTelemetry: ", t.ticks, " ticks at dt = ", t.dt, " s over ", round(t.window; digits = 2), " s",
        ismissing(t.first_sample_s) ? "" : ", first sample after $(round(t.first_sample_s; digits = 2)) s",
        t.starved ? " — STARVED (stats unreliable)" : "")
    println(io, "  columns: ", join(String.(t.columns), ", "))
    st = gpu_telemetry_stats(t)
    for k in sort!(collect(keys(st)))
        (endswith(k, "_mean") || endswith(k, "_peak")) || continue
        println(io, "  ", rpad(k, 26), round(st[k]; sigdigits = 5))
    end
    return nothing
end

# ── Tables.jl on GPUTelemetry ───────────────────────────────────────────────────────────────
#
# `GPUTelemetry` is already a column table (a `Symbol`-indexed matrix); declaring the interface
# lets DataFrames, CSV.jl, Arrow and every plotting package consume it directly.

Tables.istable(::Type{GPUTelemetry}) = true
Tables.columnaccess(::Type{GPUTelemetry}) = true
Tables.columns(t::GPUTelemetry) = t
Tables.columnnames(t::GPUTelemetry) = t.columns
Tables.getcolumn(t::GPUTelemetry, nm::Symbol) = t[nm]
Tables.getcolumn(t::GPUTelemetry, i::Int) = t.samples[:, i]
Tables.schema(t::GPUTelemetry) = Tables.Schema(t.columns, fill(Float64, length(t.columns)))
