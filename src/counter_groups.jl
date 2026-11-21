# Per-kernel tables over a multi-kernel collection, and a limiter label from speed-of-light
# counters. A window over a whole step profiles ~100 launches of a few dozen kernels; the
# question is which kernels hold the time and what limits each of them. Both functions only
# reduce what the collection holds: nothing is measured, and the label is a rule of thumb whose
# thresholds are keywords and whose inputs come back next to it.

"""    kernel_name_stem(name) -> String

The kernel's name without its parameter list, the `gpu_` prefix KernelAbstractions adds, and
trailing underscores (a Julia `!` becomes `_`): `"gpu_resort_rank_kernel_(CompilerMetadata<…>, …)"`
→ `"resort_rank_kernel"`. The default grouping of [`hw_counter_by_kernel`](@ref), which merges
the launches of one Julia kernel compiled for different argument types."""
function kernel_name_stem(name::AbstractString)
    s = strip(first(split(name, '('; limit = 2)))
    startswith(s, "gpu_") && (s = s[(nextind(s, 0, 5)):end])
    s = rstrip(s, '_')
    return isempty(s) ? String(name) : String(s)
end

# Stall reasons that are not stalls: the issuing cycle and a ready warp that lost the pick.
const _NOT_STALLS = ("selected", "not_selected")

# Time-weighted means of every derived column over the dispatches of `hcs`. The weights are the
# profiled durations when every dispatch has one, else 1 (and `time_s` is missing).
function _group_means(hcs)
    durations = [d.duration_s for h in hcs for d in h.dispatches]
    timed = !isempty(durations) && all(!ismissing, durations)
    sums = Dict{String, Float64}()
    weights = Dict{String, Float64}()
    for h in hcs
        for (k, col) in hw_counter_derived(h), (i, v) in enumerate(col)
            ismissing(v) && continue
            w = timed ? h.dispatches[i].duration_s : 1.0
            sums[k] = get(sums, k, 0.0) + w * v
            weights[k] = get(weights, k, 0.0) + w
        end
    end
    means = Dict{String, Float64}(k => sums[k] / weights[k] for k in keys(sums) if weights[k] > 0)
    return means, timed ? sum(Float64, durations) : missing
end

function _top_stalls(means::AbstractDict, n::Integer)
    stalls = Pair{String, Float64}[]
    for (r, _) in NVIDIA_STALL_REASONS
        r in _NOT_STALLS && continue
        v = get(means, "nvidia_stall_" * r, missing)
        ismissing(v) || push!(stalls, r => v)
    end
    sort!(stalls; by = last, rev = true)
    return stalls[1:min(n, length(stalls))]
end

_pct(x) = string(round(Int, 100x), "%")

function _limiter(means::AbstractDict; bound::Real = 0.6, low::Real = 0.4, occupancy_low::Real = 0.25)
    0 < low <= bound <= 1 || throw(ArgumentError("thresholds must satisfy 0 < low ≤ bound ≤ 1"))
    sm = get(means, "nvidia_sm_throughput", missing)
    mem = get(means, "nvidia_memory_throughput", missing)
    stalls = _top_stalls(means, 1)
    top = isempty(stalls) ? missing : first(first(stalls))
    stall_text = ismissing(top) ? "" : string(", top stall ", top, " ", round(last(first(stalls)); digits = 1), " cycles/issue")
    (ismissing(sm) || ismissing(mem)) && return (limiter = missing, top_stall = top,
        reason = "no speed-of-light counters: collect the :limiter preset or an ncu section set (NVIDIA only)")
    sol = string("SM ", _pct(sm), ", memory ", _pct(mem), " of peak")
    if max(sm, mem) >= bound
        return (limiter = sm >= mem ? :compute_bound : :memory_bound, top_stall = top, reason = sol * stall_text)
    elseif sm < low && mem < low
        occ = get(means, "nvidia_active_occupancy", missing)
        theo = get(means, "nvidia_theoretical_occupancy", missing)
        if !ismissing(occ) && occ < occupancy_low
            occ_text = string(", achieved occupancy ", _pct(occ), ismissing(theo) ? "" : string(" of a theoretical ", _pct(theo)))
            return (limiter = :occupancy_limited, top_stall = top, reason = sol * occ_text * stall_text)
        end
        return (limiter = :latency_bound, top_stall = top, reason = sol * stall_text)
    end
    return (limiter = :mixed, top_stall = top, reason = sol * " (between the thresholds)" * stall_text)
end

"""
    hw_counter_limiter(hc::HWCounters; bound = 0.6, low = 0.4, occupancy_low = 0.25)
    hw_counter_limiter(row; kwargs...)    # a row of hw_counter_by_kernel

A rule-of-thumb label for what limits a kernel, from NVIDIA speed-of-light counters (the
`:limiter` preset or an ncu section set), as `(limiter, top_stall, reason)`. With `sm` and
`mem` the SM and memory throughputs as fractions of their peak (time-weighted over the
dispatches), `limiter` is

- `:compute_bound` / `:memory_bound` when the larger of the two is at least `bound`;
- `:occupancy_limited` when both are below `low` and the achieved occupancy is below
  `occupancy_low`: too few warps to hide latency (theoretical occupancy capped by registers or
  shared memory, or a launch too small for the GPU);
- `:latency_bound` when both are below `low` otherwise: warps wait (`top_stall` says on what,
  `long_scoreboard` being memory latency);
- `:mixed` in between, and `missing` when the throughputs were not collected (AMD has no rules
  yet).

`reason` quotes the numbers the label came from. It is a starting point, not a diagnosis:
read the stall reasons and the throughputs before acting on it. Throughputs are ratios to the
peak at the clock the kernel ran at, so the label survives `clock_control = :base`.
"""
hw_counter_limiter(h::HWCounters; kwargs...) = _limiter(first(_group_means([h])); kwargs...)
function hw_counter_limiter(row::NamedTuple; kwargs...)
    means = Dict{String, Float64}(String(k) => v for (k, v) in pairs(row) if v isa Real && !(v isa Bool))
    return _limiter(means; kwargs...)
end

"""
    hw_counter_by_kernel(path; group = kernel_name_stem, top_stalls = 3, limiter = (;), kwargs...)
    hw_counter_by_kernel(hcs::AbstractVector{HWCounters}; group, top_stalls, limiter)

One row per kernel group of a multi-kernel collection (see [`hw_counter_kernels`](@ref), whose
keywords `path` takes), ordered by profiled time, as a vector of named tuples (a Tables.jl
table). `group` maps a full kernel name to its group; the default
[`kernel_name_stem`](@ref) merges one kernel's launches across argument types. Columns:

- `group`, `kernels` (the full names), `dispatches`;
- `time_s`, the summed profiled duration, and `share`, its fraction of the total over the
  kernels selected (AMD blit kernels and launches outside a `kernel` filter excluded);
- one column per derived column of [`hw_counter_derived`](@ref) present in any group, as the
  mean over the group's dispatches weighted by their durations (`missing` where the group
  lacks it);
- `top_stalls`, the `top_stalls` largest NVIDIA stall reasons as `reason => cycles per issued
  instruction` (`selected` and `not_selected` excluded), and `limiter` / `limiter_reason` from
  [`hw_counter_limiter`](@ref) with the `limiter` keywords.

Durations under replay (ncu) or with `clock_control = :base` are not the run's timings: the
shares are comparable within the collection, the seconds are not comparable with a timed
run. A group without a duration on every dispatch has `time_s = missing` and unweighted means,
and then every row's `share` is `missing` (the total is unknown). Stall ratios are weighted by
duration, not by issued instructions, so a group's `top_stalls` ranks where its time went.
"""
function hw_counter_by_kernel(path::AbstractString; group = kernel_name_stem, top_stalls::Integer = 3,
        limiter = (;), kwargs...)
    return hw_counter_by_kernel(hw_counter_kernels(path; kwargs...); group, top_stalls, limiter)
end
function hw_counter_by_kernel(hcs::AbstractVector{HWCounters}; group = kernel_name_stem,
        top_stalls::Integer = 3, limiter = (;))
    top_stalls >= 0 || throw(ArgumentError("top_stalls must be non-negative"))
    groups = Dict{String, Vector{HWCounters}}()
    order = String[]
    for h in hcs
        g = String(group(h.kernel))
        haskey(groups, g) || push!(order, g)
        push!(get!(() -> HWCounters[], groups, g), h)
    end
    reduced = [(g, _group_means(groups[g])...) for g in order]
    columns = sort!(unique(k for (_, means, _) in reduced for k in keys(means)))
    total = any(r -> ismissing(r[3]), reduced) ? missing : sum(Float64, (r[3] for r in reduced); init = 0.0)
    names = (:group, :kernels, :dispatches, :time_s, :share, Symbol.(columns)..., :top_stalls, :limiter, :limiter_reason)
    M = Union{Missing, Float64}
    NT = NamedTuple{names, Tuple{String, Vector{String}, Int, M, M, ntuple(_ -> M, length(columns))...,
        Vector{Pair{String, Float64}}, Union{Missing, Symbol}, String}}
    rows = NT[]
    for (g, means, t) in reduced
        lim = _limiter(means; limiter...)
        share = ismissing(t) || ismissing(total) || total <= 0 ? missing : t / total
        push!(rows, NT((g, [h.kernel for h in groups[g]], sum(length, groups[g]), t, share,
            (get(means, c, missing) for c in columns)..., _top_stalls(means, top_stalls), lim.limiter, lim.reason)))
    end
    sort!(rows; by = r -> coalesce(r.time_s, -Inf), rev = true)
    return rows
end
