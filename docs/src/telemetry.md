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
| `throttle_reasons` | bitmask | `NaN` (the `gpu_metrics` blob is not decoded yet) | `nvmlDeviceGetCurrentClocksEventReasons` |

`throttle_reasons(x)` decodes the bitmask into NVML's names (`:gpu_idle`, `:sw_power_cap`,
`:hw_slowdown`, `:sw_thermal_slowdown`, `:hw_thermal_slowdown`, `:hw_power_brake_slowdown`, …).

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
