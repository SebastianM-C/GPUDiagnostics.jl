#!/usr/bin/env python3
"""Regenerate the `_GPU_METRICS_LAYOUTS` table in src/amd_gpu_metrics.jl from the Linux kernel's
kgd_pp_interface.h (struct gpu_metrics_v*). Run when the kernel adds a revision:

    curl -sL https://raw.githubusercontent.com/torvalds/linux/master/drivers/gpu/drm/amd/include/kgd_pp_interface.h > /tmp/kgd.h
    python3 tools/gen_gpu_metrics_layouts.py /tmp/kgd.h > /tmp/layouts.jl   # then paste over the const

Offsets follow the C ABI (natural alignment, struct padded to its widest member); a second entry
per revision gives the packed size, in case a driver emits the struct without implicit padding —
the header's structure_size selects at run time, and a size matching neither yields `missing`."""
import re, sys, pathlib

src = pathlib.Path(sys.argv[1]).read_text()
consts = {m.group(1): int(m.group(2)) for m in re.finditer(r"#define\s+(NUM_\w+|MAX_\w+)\s+(\d+)", src)}
sizes = {"uint8_t": 1, "uint16_t": 2, "uint32_t": 4, "uint64_t": 8, "int8_t": 1, "int16_t": 2, "int32_t": 4, "int64_t": 8}
# kernel field name -> the decoder's key
want = {"throttle_status": "throttle_status", "indep_throttle_status": "indep_throttle_status",
        "average_gfx_activity": "gfx_activity", "average_umc_activity": "umc_activity",
        "average_socket_power": "socket_power_W", "curr_socket_power": "socket_power_W",
        "current_gfxclk": "gfxclk_MHz", "temperature_hotspot": "hotspot_C", "temperature_mem": "mem_temperature_C",
        "accumulation_counter": "accumulation_counter", "prochot_residency_acc": "prochot_residency_acc",
        "ppt_residency_acc": "ppt_residency_acc", "socket_thm_residency_acc": "socket_thm_residency_acc",
        "vr_thm_residency_acc": "vr_thm_residency_acc", "hbm_thm_residency_acc": "hbm_thm_residency_acc"}

def layout(name, aligned):
    body = re.search(r"struct %s \{(.*?)\n\};" % name, src, re.S).group(1)
    off, fields, maxal = 0, [], 2
    for line in body.splitlines():
        line = line.split("//")[0].split("/*")[0].strip()
        if not line or line.startswith("*"):
            continue
        if line.startswith("struct metrics_table_header"):
            off += 4
            continue
        mm = re.match(r"(\w+)\s+(\w+)(?:\[(\w+)\])?(?:\[(\w+)\])?;", line)
        if not mm:
            continue
        t, n, a1, a2 = mm.groups()
        sz, cnt = sizes[t], 1
        for a in (a1, a2):
            if a:
                cnt *= consts.get(a, int(a) if a.isdigit() else 0)
        if aligned and off % sz:
            off += sz - off % sz
        maxal = max(maxal, sz)
        fields.append((n, off, sz, cnt))
        off += sz * cnt
    if aligned and off % maxal:
        off += maxal - off % maxal
    return off, fields

versions = [f"v1_{i}" for i in range(0, 9)] + [f"v2_{i}" for i in range(0, 5)]
print("const _GPU_METRICS_LAYOUTS = Dict{Tuple{Int, Int, Int}, Dict{Symbol, Tuple{Int, Int, Int}}}(")
seen = set()
for v in versions:
    fmt, con = map(int, v[1:].split("_"))
    for aligned in (True, False):
        size, f = layout("gpu_metrics_" + v, aligned)
        if (fmt, con, size) in seen:
            continue
        seen.add((fmt, con, size))
        entries = ", ".join(f":{want[n]} => ({o}, {sz}, {c})" for n, o, sz, c in f if n in want)
        print(f"    ({fmt}, {con}, {size}) => Dict({entries}),")
print(")")
