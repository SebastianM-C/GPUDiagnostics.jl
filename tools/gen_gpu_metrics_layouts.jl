#!/usr/bin/env julia
# Regenerate the `_GPU_METRICS_LAYOUTS` table in src/amd_gpu_metrics.jl from the Linux kernel's
# kgd_pp_interface.h (struct gpu_metrics_v*). Run when the kernel adds a revision:
#
#     curl -sL https://raw.githubusercontent.com/torvalds/linux/master/drivers/gpu/drm/amd/include/kgd_pp_interface.h > /tmp/kgd.h
#     julia tools/gen_gpu_metrics_layouts.jl /tmp/kgd.h > /tmp/layouts.jl   # then paste over the const
#
# Offsets follow the C ABI (natural alignment, struct padded to its widest member); a second entry
# per revision gives the packed size, in case a driver emits the struct without implicit padding —
# the header's structure_size selects at run time, and a size matching neither yields `missing`.
# Plain Julia, no packages.

const SIZES = Dict("uint8_t" => 1, "uint16_t" => 2, "uint32_t" => 4, "uint64_t" => 8,
    "int8_t" => 1, "int16_t" => 2, "int32_t" => 4, "int64_t" => 8)
# kernel field name => the decoder's key
const WANT = Dict("throttle_status" => "throttle_status", "indep_throttle_status" => "indep_throttle_status",
    "average_gfx_activity" => "gfx_activity", "average_umc_activity" => "umc_activity",
    "average_socket_power" => "socket_power_W", "curr_socket_power" => "socket_power_W",
    "current_gfxclk" => "gfxclk_MHz", "temperature_hotspot" => "hotspot_C", "temperature_mem" => "mem_temperature_C",
    "accumulation_counter" => "accumulation_counter", "prochot_residency_acc" => "prochot_residency_acc",
    "ppt_residency_acc" => "ppt_residency_acc", "socket_thm_residency_acc" => "socket_thm_residency_acc",
    "vr_thm_residency_acc" => "vr_thm_residency_acc", "hbm_thm_residency_acc" => "hbm_thm_residency_acc")

function layout(src, consts, name; aligned)
    m = match(Regex("struct $name \\{(.*?)\\n\\};", "s"), src)
    m === nothing && return nothing
    off, maxal = 0, 2
    fields = Tuple{String, Int, Int, Int}[]
    for line in split(m[1], '\n')
        line = strip(first(split(first(split(line, "//")), "/*")))
        (isempty(line) || startswith(line, "*")) && continue
        if startswith(line, "struct metrics_table_header")
            off += 4
            continue
        end
        f = match(r"^(\w+)\s+(\w+)(?:\[(\w+)\])?(?:\[(\w+)\])?;", line)
        f === nothing && continue
        t, n, a1, a2 = f.captures
        sz = SIZES[t]
        cnt = 1
        for a in (a1, a2)
            a === nothing && continue
            cnt *= get(consts, a, all(isdigit, a) ? parse(Int, a) : 0)
        end
        aligned && off % sz != 0 && (off += sz - off % sz)
        maxal = max(maxal, sz)
        push!(fields, (String(n), off, sz, cnt))
        off += sz * cnt
    end
    aligned && off % maxal != 0 && (off += maxal - off % maxal)
    return off, fields
end

function main(path)
    src = read(path, String)
    consts = Dict(m[1] => parse(Int, m[2]) for m in eachmatch(r"#define\s+((?:NUM|MAX)_\w+)\s+(\d+)", src))
    println("const _GPU_METRICS_LAYOUTS = Dict{Tuple{Int, Int, Int}, Dict{Symbol, Tuple{Int, Int, Int}}}(")
    seen = Set{Tuple{Int, Int, Int}}()
    revisions = vcat([(1, i) for i in 0:8], [(2, i) for i in 0:4])
    for (fmt, con) in revisions, aligned in (true, false)
        r = layout(src, consts, "gpu_metrics_v$(fmt)_$(con)"; aligned)
        r === nothing && continue
        size, fields = r
        (fmt, con, size) in seen && continue
        push!(seen, (fmt, con, size))
        entries = join([":$(WANT[n]) => ($o, $sz, $c)" for (n, o, sz, c) in fields if haskey(WANT, n)], ", ")
        println("    ($fmt, $con, $size) => Dict($entries),")
    end
    println(")")
end

length(ARGS) == 1 || (println(stderr, "usage: julia tools/gen_gpu_metrics_layouts.jl <kgd_pp_interface.h>"); exit(64))
main(ARGS[1])
