# Telemetry

```julia
gpu_sample(backend, 1)              # NamedTuple: the base columns below (+ GPM counters on NVIDIA)

result, telem = with_gpu_sampler(backend, 1.0; devices = 1:2, tracefile = "gputrace.tsv") do
    run_the_workload()
end
telem.columns                       # [:t_rel_s, :device, base columns…, (:sm_util, :sm_occupancy, :fp64_util, …)]
telem[:compute_util]                # a column; length(telem) rows over all devices
gpu_telemetry_stats(telem)          # "<col>_mean" / "_peak" / "_busy_mean" / "_busy_median" (rows with compute_util ≥ 0.5),
                                    # "samples", "busy_samples", "power_capped_fraction" (busy rows at power_W ≥ 0.95 × power_limit_W)
throttle_reasons(telem[:throttle_reasons][end])   # e.g. [:sw_power_cap, :sw_thermal_slowdown]
```

Base columns, every source, `NaN` where the vendor or device has none:

| column | unit | AMD (amdgpu sysfs) | NVIDIA (NVML) |
|---|---|---|---|
| `power_W` | W | hwmon `power1_average` / `power1_input` | `nvmlDeviceGetPowerUsage` |
| `compute_util`, `mem_util` | fraction | `gpu_busy_percent`, `mem_busy_percent` | `nvmlDeviceGetUtilizationRates` |
| `vram_used_B` | B | `mem_info_vram_used` | `nvmlDeviceGetMemoryInfo` |
| `sm_clock_MHz`, `mem_clock_MHz` | MHz | hwmon `freq1_input`, `freq2_input` | `nvmlDeviceGetClockInfo` (SM, MEM) |
| `temperature_C` | °C | hwmon `temp1_input` (edge) | `nvmlDeviceGetTemperature` (GPU) |
| `hotspot_C` | °C | hwmon `temp2_input` (junction) | `NaN` (no NVML junction sensor) |
| `power_limit_W` | W | hwmon `power1_cap` | `nvmlDeviceGetEnforcedPowerLimit` |
| `throttle_reasons` | bitmask | `NaN` (NVML's bit layout; AMD's is `amd_throttle_status` below) | `nvmlDeviceGetCurrentClocksEventReasons` |

`throttle_reasons(x)` decodes the bitmask into NVML's names (`:gpu_idle`, `:sw_power_cap`,
`:hw_slowdown`, `:sw_thermal_slowdown`, `:hw_thermal_slowdown`, `:hw_power_brake_slowdown`, …).

**AMD throttler state (the `gpu_metrics` blob).** The amdgpu driver keeps the SMU's throttler
state out of hwmon and in one binary sysfs snapshot, `gpu_metrics`, whose versioned layouts the
kernel publishes in `kgd_pp_interface.h`. The sampler child reads it each tick next to the hwmon
files, and the columns below appear on AMD devices whose driver exposes the file (`NaN` where a
revision lacks the field; NVIDIA rows never carry them):

| column | unit | source | revisions |
|---|---|---|---|
| `amd_throttle_status` | bitmask | `indep_throttle_status`, the driver's vendor-independent throttler bits (`amd_throttle_reasons(x)` names them: `:ppt0`…, `:tdc_*`, `:temp_hotspot`, …); the raw SMU mask where a revision has only that | v1.3 (RDNA3 dGPUs), v2.2+ (APUs); raw on v1.0–1.5 |
| `xcd_clock_min_MHz`, `xcd_clock_max_MHz` | MHz | spread of `current_gfxclk[]` over the dies | v1.4+ (MI300 and later); a single die gives min = max |
| `throttle_acc_counter`, `power_throttle_acc`, `thermal_throttle_acc`, `hbm_throttle_acc`, `vr_throttle_acc`, `prochot_acc` | counts | the accumulated throttler residencies: PPT (package power), socket thermal, HBM thermal, VR thermal, PROCHOT, with their accumulation counter | v1.6+ (MI300) |

The residencies are cumulative, so `gpu_telemetry_stats` differences them over the window:
`power_violation_fraction` is Δ`power_throttle_acc` / Δ`throttle_acc_counter`, AMD's own PVIOL
definition (AMD SMI's `amd-smi metric --violation` reports the same numbers on bare metal), and
likewise `thermal_violation_fraction`, `hbm_thermal_violation_fraction`,
`vr_thermal_violation_fraction`, `prochot_fraction`. That is the direct reading of "held at the
power cap" against "held by a temperature", which `power_capped_fraction` (power within 5 % of
the cap) can only infer. From the bitmasks, on either vendor, `power_throttled_fraction` and
`thermal_throttled_fraction` are the share of busy rows in which a power or a thermal limiter
was reported.

Two caveats. A W7900 (RDNA3, v1.3) reports the `:temp_hotspot` bit permanently — at 9 W and a
3 MHz clock with the junction at 38 °C, and unchanged through a 12 s FP32 burn at 81 °C — so on
RDNA3 `thermal_throttled_fraction` reads 1 whatever the temperature and only the power bits carry
information. Those do behave: the same burn held 240 W against the 241 W cap and reported `:ppt0`
on 55 of 56 busy samples, `power_throttled_fraction` = `power_capped_fraction` = 1.0, the shader
clock at 2.74 GHz. And whether an SR-IOV
virtual function (a cloud MI300X) exposes `gpu_metrics` at all is up to the host; the columns
are simply absent when the file is. `amd_gpu_metrics(path)` decodes one snapshot offline for
inspection; the field offsets are data, `assets/gpu_metrics_layouts.toml`, which
`tools/gen_gpu_metrics_layouts.jl` rewrites from the kernel header when a new revision appears.

`gpu_sample` is exactly what the sampler child calls per tick. The child is a Julia process
(`telemetry_child_main`, started with the parent's julia binary and load path) rather than a Julia
task: an in-process tick either wedges on the vendor runtime behind a backed-up kernel stream, or
is suspended with the sleeping task by Julia's GC/timer coupling while the host thread allocates.
The vendor extension resolves per-device *source specs* in the parent (`gpu_sampler_sources`) so the
child needs no vendor runtime: on AMD it reads the amdgpu driver's sysfs files it was handed and
starts in about a second; on NVIDIA it loads CUDA.jl for its NVML bindings only (no CUDA context,
a few seconds of startup that `telem.first_sample_s` records and the starvation watchdog discounts).
The child declares its own column header, so the parent parses whatever metric set it emits; it
appends rows as it samples (the trace survives a crash) and stops cooperatively via a stopfile, or
on its own if the parent dies. Any failure to build or start the child logs a warning and the
function runs unsampled.

**GPM counters (NVIDIA Hopper and newer).** With `counters = :auto` (default), devices that support
GPU Performance Monitoring — H100 / H200 / GH200 / B200 and, with recent drivers, consumer Blackwell
(RTX 5090 on driver 580 verified); no profiling privileges needed — add ACHIEVED SM occupancy (the
number to hold against the compile-time theoretical occupancy of `kernel_resources`), FP64 / FP32 /
FP16 / tensor / integer pipe utilization, DRAM-bandwidth utilization and PCIe / NVLink traffic, each
averaged over the interval between two consecutive ticks. `counters = :none` skips them. Note that
`fp64_util` is normalised to the SM's full-rate issue slots: a saturated FP64 FMA chain reads ≈ 0.9
on an H100 but only ≈ 1.5 % on a 1/64-rate consumer board.
