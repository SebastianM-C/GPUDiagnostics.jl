# The amdgpu driver's `gpu_metrics` sysfs file: one binary snapshot of the SMU's view of the
# device — activity, clocks, power, temperatures and, above all, the THROTTLER STATE, which no
# hwmon file carries. The kernel documents the file (Documentation/gpu/amdgpu/thermal.rst) and
# publishes the layouts as versioned structs in drivers/gpu/drm/amd/include/kgd_pp_interface.h;
# a 4-byte header (structure_size u16, format_revision u8, content_revision u8) says which.
#
# Two families matter:
# - v1.3 (RDNA3 dGPUs such as the W7900) and v2.2+ (APUs): an instantaneous bitmask.
#   `throttle_status` is the SMU's own bit layout; `indep_throttle_status` is the driver's
#   vendor-independent remap (SMU_THROTTLER_*_BIT in drivers/gpu/drm/amd/pm/swsmu/inc/amdgpu_smu.h),
#   which is what `amd_throttle_reasons` decodes.
# - v1.6+ (MI300 and later): no bitmask, but ACCUMULATED THROTTLER RESIDENCIES — counters that
#   advance while the package-power (PPT), socket-thermal, VR-thermal, HBM-thermal or PROCHOT
#   limiter is active — next to an `accumulation_counter`. Two reads a known interval apart give
#   the fraction of that interval spent under each limiter: AMD's own definition (AMD SMI,
#   `amdsmi_gpu_metrics_t`) is PVIOL % = Δppt_residency_acc × 100 / Δaccumulation_counter. That is
#   the direct answer to "is this kernel power-bound", separately from "thermally bound". v1.4+
#   also carry the per-XCD gfx clocks (`current_gfxclk[8]`), so die-to-die clock spread is visible.
#
# AMD SMI (`amd-smi metric --violation`, `amdsmi_get_violation_status`) decodes the same data but
# is tagged bare-metal-Linux only and needs the library plus device-node access; reading the
# sysfs file needs neither, works inside a container, and is what the sampler child already does
# for hwmon. Whether an SR-IOV virtual function exposes the file at all is per host.
#
# Nothing here is defaulted: an unknown revision, a size that matches no known layout, or an
# absent field yields `missing` (and NaN in the telemetry), never a guess.

# The field offsets per (format, content, structure_size) live in assets/gpu_metrics_layouts.toml,
# written from the kernel header by tools/gen_gpu_metrics_layouts.jl and baked in at precompile
# time. Each entry: field => (byte offset, width in bytes, count).
const _GPU_METRICS_LAYOUTS_FILE = joinpath(@__DIR__, "..", "assets", "gpu_metrics_layouts.toml")
include_dependency(_GPU_METRICS_LAYOUTS_FILE)
function _load_gpu_metrics_layouts(path::AbstractString)
    out = Dict{Tuple{Int, Int, Int}, Dict{Symbol, Tuple{Int, Int, Int}}}()
    for l in TOML.parsefile(path)["layout"]
        out[(Int(l["format"]), Int(l["content"]), Int(l["size"]))] =
            Dict(Symbol(k) => (Int(v[1]), Int(v[2]), Int(v[3])) for (k, v) in l["fields"])
    end
    return out
end
const _GPU_METRICS_LAYOUTS = _load_gpu_metrics_layouts(_GPU_METRICS_LAYOUTS_FILE)

# The driver's vendor-independent throttler bits (amdgpu_smu.h SMU_THROTTLER_*_BIT), in bit order.
# Bits 0–7 are power limiters, 16–23 current/EDC limiters, 32–47 thermal and PROCHOT, 56–57 other.
const AMD_THROTTLER_BITS = (
    (0, :ppt0), (1, :ppt1), (2, :ppt2), (3, :ppt3), (4, :spl), (5, :fppt), (6, :sppt), (7, :sppt_apu),
    (16, :tdc_gfx), (17, :tdc_soc), (18, :tdc_mem), (19, :tdc_vdd), (20, :tdc_cvip), (21, :edc_cpu),
    (22, :edc_gfx), (23, :apcc),
    (32, :temp_gpu), (33, :temp_core), (34, :temp_mem), (35, :temp_edge), (36, :temp_hotspot),
    (37, :temp_soc), (38, :temp_vr_gfx), (39, :temp_vr_soc), (40, :temp_vr_mem0), (41, :temp_vr_mem1),
    (42, :temp_liquid0), (43, :temp_liquid1), (44, :vrhot0), (45, :vrhot1), (46, :prochot_cpu),
    (47, :prochot_gfx), (56, :ppm), (57, :fit),
)
# Which of those bits mean "held back by the power budget" vs "by a temperature": the two
# fractions the stats derive from a bitmask column, on either vendor.
const _AMD_POWER_BITS = UInt64(0xff) | (UInt64(0xff) << 16)          # PPT/SPL/FPPT/SPPT and the current limiters
const _AMD_THERMAL_BITS = UInt64(0xffff) << 32                         # TEMP_*, VRHOT, PROCHOT
const _NVML_POWER_BITS = UInt64(0x004 | 0x080)                         # sw_power_cap, hw_power_brake_slowdown
const _NVML_THERMAL_BITS = UInt64(0x020 | 0x040)                       # sw/hw thermal slowdown

"""    amd_throttle_reasons(x::Real) -> Vector{Symbol}

Decode an `amd_throttle_status` sample — the amdgpu driver's vendor-independent throttler
bitmask (`indep_throttle_status` of the `gpu_metrics` blob), stored in the telemetry as an
integer-valued `Float64` — into the kernel's names: `:ppt0`…`:ppt3`, `:spl`, `:fppt`, `:sppt`
(power limiters), `:tdc_*`, `:edc_*`, `:apcc` (current), `:temp_gpu`, `:temp_hotspot`,
`:temp_mem`, `:temp_vr_*`, `:vrhot*`, `:prochot_*` (thermal), `:ppm`, `:fit`. `NaN` and 0 give an
empty vector; an unknown bit is `:bit_<n>`. The NVML counterpart is [`throttle_reasons`](@ref).
Known quirk: an idle RDNA3 W7900 reports `:temp_hotspot` permanently (see the telemetry guide),
so read this column under load and against the temperatures, not in isolation.
"""
function amd_throttle_reasons(x::Real)
    out = Symbol[]
    (x isa AbstractFloat && !isfinite(x)) && return out
    isinteger(x) && x >= 0 || throw(ArgumentError("amd_throttle_reasons: expected a non-negative integer bitmask, got $x"))
    mask = UInt64(x)
    for (bit, name) in AMD_THROTTLER_BITS
        mask & (UInt64(1) << bit) == 0 || push!(out, name)
        mask &= ~(UInt64(1) << bit)
    end
    while mask != 0
        n = trailing_zeros(mask)
        push!(out, Symbol("bit_", n))
        mask &= ~(UInt64(1) << n)
    end
    return out
end

"""    amd_gpu_metrics(path) -> NamedTuple

Decode one `gpu_metrics` snapshot (a path, or the raw bytes) into `version::VersionNumber`,
`structure_size`, and — each `missing` when the revision lacks it — `throttle_status`,
`indep_throttle_status` (both `UInt64`), `gfx_activity` and `umc_activity` (fractions),
`socket_power_W`, `hotspot_C`, `mem_temperature_C`, `gfxclk_MHz` (a vector: one entry per XCD
on MI300-class parts, one on single-die GPUs; unpopulated slots removed), and the MI300
residency accumulators `accumulation_counter`, `ppt_residency_acc`, `socket_thm_residency_acc`,
`vr_thm_residency_acc`, `hbm_thm_residency_acc`, `prochot_residency_acc` (raw counters; a
fraction is Δresidency / Δaccumulation_counter between two snapshots). A revision or size this
package does not know yields the header fields and `missing` everywhere else. Pure Julia; no
ROCm library, no device access beyond reading the file.
"""
amd_gpu_metrics(path::AbstractString) = amd_gpu_metrics(read(path))
function amd_gpu_metrics(bytes::AbstractVector{UInt8})
    length(bytes) >= 4 || throw(ArgumentError("gpu_metrics: $(length(bytes)) bytes, need at least the 4-byte header"))
    size = Int(reinterpret(UInt16, bytes[1:2])[1])
    fmt, con = Int(bytes[3]), Int(bytes[4])
    layout = get(_GPU_METRICS_LAYOUTS, (fmt, con, size), nothing)
    rd(key) = begin
        layout === nothing && return missing
        e = get(layout, key, nothing)
        e === nothing && return missing
        off, w, n = e
        off + w * n <= length(bytes) || return missing
        T = w == 1 ? UInt8 : w == 2 ? UInt16 : w == 4 ? UInt32 : UInt64
        vals = [T(reinterpret(T, bytes[(off + 1 + w * (i - 1)):(off + w * i)])[1]) for i in 1:n]
        n == 1 ? vals[1] : vals
    end
    invalid(T) = typemax(T)   # the SMU writes all-ones for a sensor it does not have
    scalar(key, T, scale) = (v = rd(key); ismissing(v) || v == invalid(T) ? missing : Float64(v) * scale)
    clks = rd(:gfxclk_MHz)
    gfxclk = ismissing(clks) ? Float64[] :
        Float64[c for c in (clks isa AbstractVector ? clks : [clks]) if c != 0 && c != invalid(UInt16)]
    mask(key) = (v = rd(key); ismissing(v) ? missing : UInt64(v))
    acc(key) = (v = rd(key); ismissing(v) || v == invalid(UInt32) ? missing : UInt32(v))
    return (version = VersionNumber(fmt, con), structure_size = size, known = layout !== nothing,
        throttle_status = mask(:throttle_status), indep_throttle_status = mask(:indep_throttle_status),
        gfx_activity = scalar(:gfx_activity, UInt16, 0.01), umc_activity = scalar(:umc_activity, UInt16, 0.01),
        socket_power_W = scalar(:socket_power_W, UInt16, 1.0), hotspot_C = scalar(:hotspot_C, UInt16, 1.0),
        mem_temperature_C = scalar(:mem_temperature_C, UInt16, 1.0), gfxclk_MHz = gfxclk,
        accumulation_counter = acc(:accumulation_counter), ppt_residency_acc = acc(:ppt_residency_acc),
        socket_thm_residency_acc = acc(:socket_thm_residency_acc), vr_thm_residency_acc = acc(:vr_thm_residency_acc),
        hbm_thm_residency_acc = acc(:hbm_thm_residency_acc), prochot_residency_acc = acc(:prochot_residency_acc))
end

# The sampler's view: the telemetry columns one snapshot contributes. `amd_throttle_status` is the
# independent bitmask (v1.3 / v2.2+; the raw SMU mask where that is all a revision has), the
# `*_acc` columns are the MI300 residency counters (cumulative — the stats take differences), and
# the per-XCD clock spread is reduced to its min and max. NaN where the revision has no such field.
const _GPU_METRICS_COLUMNS = (:amd_throttle_status, :xcd_clock_min_MHz, :xcd_clock_max_MHz,
    :throttle_acc_counter, :power_throttle_acc, :thermal_throttle_acc, :hbm_throttle_acc,
    :vr_throttle_acc, :prochot_acc)
_gpu_metrics_nan() = NamedTuple{_GPU_METRICS_COLUMNS}(ntuple(_ -> NaN, length(_GPU_METRICS_COLUMNS)))
function _gpu_metrics_columns(path)
    path === nothing && return _gpu_metrics_nan()
    m = try
        amd_gpu_metrics(path)
    catch
        return _gpu_metrics_nan()   # transient read failure or a truncated file → NaN this tick
    end
    f(x) = ismissing(x) ? NaN : Float64(x)
    bits = ismissing(m.indep_throttle_status) ? m.throttle_status : m.indep_throttle_status
    return (amd_throttle_status = f(bits),
        xcd_clock_min_MHz = isempty(m.gfxclk_MHz) ? NaN : minimum(m.gfxclk_MHz),
        xcd_clock_max_MHz = isempty(m.gfxclk_MHz) ? NaN : maximum(m.gfxclk_MHz),
        throttle_acc_counter = f(m.accumulation_counter), power_throttle_acc = f(m.ppt_residency_acc),
        thermal_throttle_acc = f(m.socket_thm_residency_acc), hbm_throttle_acc = f(m.hbm_thm_residency_acc),
        vr_throttle_acc = f(m.vr_thm_residency_acc), prochot_acc = f(m.prochot_residency_acc))
end
