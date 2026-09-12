# Hardware counters

Counters answer *what the kernel did*: instructions issued, waves resident, cache requests,
cycles. They do not answer *how long it took*; that is [`LaunchTimer`](device_timing.md)'s job,
with [`with_gpu_sampler`](telemetry.md) alongside for clocks and power. A performance study
therefore runs the same kernel twice, and the split is the same on every vendor:

1. **The timing run**, unprofiled, in the card's normal power state: `LaunchTimer` medians over
   warm launches. This is the benchmark.
2. **The counter run**, a few dispatches of the same kernel under the profiler, built with
   `hw_counter_command`. The profiler changes how the kernel runs: Nsight Compute replays and
   serialises launches, RDNA GPUs need a fixed performance level before some counters read at
   all, and only CDNA parts profiled within a few percent of their event timings.

Two kinds of values come out of the counter run. **Counts** — instructions per slot, waves,
requests, hit rates, wait fractions — are exact whatever the clock did, and are the code's
behaviour as run. **Anything divided by a duration** — the active clock, achieved bandwidth,
the profiled duration itself — describes the clock state during the collection, which on an
RDNA card pinned to its standard level or under `ncu`'s replay is not the state of the timing
run. The join between the two runs is the cycle count: every preset carries `GRBM_GUI_ACTIVE`
(or `gpu__time_duration` with the SM clock) because cycles per slot is the cost of the code
independent of clock, and the timing run's duration turns it into the clock the kernel really
ran at. That is also how a power-managed part is read: the same work in fewer cycles is a
better kernel, wall time falling less than the cycles is the clock dropping.

Per-dispatch counters come from rocprofv3 on AMD and Nsight Compute (`ncu`) on NVIDIA.
Both produce `HWCounters`. Sampled counters, such as NVML GPM, remain in `GPUTelemetry`;
they measure a different window and have a separate capability.

```julia
using GPUDiagnostics, CUDA, KernelAbstractions

backend = CUDABackend()              # use ROCBackend() with AMDGPU for AMD
mkpath("prof")
cmd = hw_counter_command(backend, `julia --project run.jl`;
    set = :fp64, dir = "prof", name = "cell",
    kernel = "my_kernel", launch_skip = 1, launch_count = 2)
run(cmd)
hc = hw_counters("prof"; name = "cell", kernel = "my_kernel",
    slots = n_work_items * n_iterations_per_item,
    required_metrics = COUNTER_SETS[:nvidia][:fp64].metrics,
    provenance = Dict("counter_set" => "fp64", "clock_control" => "none",
        "cache_control" => "all", "replay_mode" => "kernel"))
hc.dispatches                       # identity, device, resources and duration of each launch
hw_counter_derived(hc)               # vectors aligned with those dispatches
hw_counter_summary(hc)               # medians, ranges and valid sample counts
diagnostics_dict(hc; prefix = "hw_") # flat, schema-tagged, TOML-safe summary
```

The backend's extension selects the profiler: `CUDABackend()` uses Nsight Compute and
`ROCBackend()` uses rocprofv3. The same intent names apply to both. Pass the backend
to `hw_counter_status` and `hw_counters_available` as well. Loading both extensions
does not introduce a global default; the backend argument determines the collector.
NVIDIA launch-skip/count are collector options; omit them for AMD and bound its workloads
in the script itself. Other collector-specific keywords belong only to that collector.
`hw_counter_command` executes nothing and creates no directories. It preserves the
workload's environment and working directory; relative output paths resolve there.

When preparing an external workload without loading a GPU runtime, select the collector
explicitly with `hw_counter_command(NsightCompute(), cmd; ...)` or
`hw_counter_command(RocprofV3(), cmd; ...)`. These collector types also work with tool
discovery. Vendor symbols remain available for preset lookup and optional CSV parser
selection; collection commands use backend or collector dispatch.

## Collection requirements

The following setup guidance targets Linux collection. Parsing existing CSV files requires
neither a GPU nor a profiler installation. Install the collector on the machine that runs
the workload, and use a profiler release that supports its GPU, driver, and operating system.
Installing CUDA.jl or AMDGPU.jl alone does not establish that the collector is installed or
that counter access is enabled.

| Requirement | NVIDIA | AMD |
|---|---|---|
| Collector | Nsight Compute CLI, `ncu` | ROCprofiler-SDK CLI, `rocprofv3` |
| Runtime | Working CUDA workload and a compatible NVIDIA driver | Working HIP/HSA workload with profiler registration enabled |
| Access | GPU access **and** driver permission to profile counters | ROCm device access **and**, when needed, permission to configure the performance state |
| Metrics | Full metric names supported by this GPU and ncu release | Counters supported by this architecture and rocprofv3 release |
| Output | Writable collection directory; CSV or explicit `.ncu-rep` export | Writable collection directory; CSV output enabled |

Put the collector on `PATH`, or pass `executable = "/path/to/collector"` to both
`hw_counter_status` and `hw_counter_command`. The command builder also uses `timeout`
by default; set `timeout_s = nothing` if that utility is unavailable and bound the workload
yourself.

`supports(backend, :hw_counters)` means an implementation exists. `hw_counters_available`
checks tool discovery only. `hw_counter_status` exposes the executable and leaves
`permitted = missing`: PATH discovery cannot establish profiling permissions. Collector
failures remain failures of `run(cmd)`, including access denied. Some tool versions
silently omit unknown metrics: `required_metrics` requires finite values for each listed
counter on every selected dispatch and detects such incomplete collections.
Neither tool is a dependency of the package or its CSV parsers.

### NVIDIA: installation and profiling permission

Install [Nsight Compute](https://developer.nvidia.com/tools-overview/nsight-compute/get-started)
and check its release-specific [GPU/platform support](https://docs.nvidia.com/nsight-compute/ReleaseNotes/topics/platform-support.html)
and [driver requirements](https://docs.nvidia.com/nsight-compute/ReleaseNotes/topics/system-requirements.html).
A working CUDA kernel or `nvidia-smi` is insufficient to establish counter access.

```sh
ncu --version
ncu --query-metrics --query-metrics-mode all
```

The second command lists full metric names, including suffixes such as `.sum`, for the
devices visible to ncu. Use names from that list; enumeration does not prove collection
permission. See the [ncu CLI reference](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html).

`ERR_NVGPUCTRPERM` means the driver denied profiling access. An administrator can grant
non-admin profiling access or arrange an appropriately privileged profiling process.
The mechanism depends on the driver release:

- Legacy Linux configuration uses `NVreg_RestrictProfilingToAdminUsers`. Setting it to
  `0` permits non-admin profiling; applying the change can require driver reload or reboot.
- R610+ drivers also support access grants through NVIDIA profiling capability device
  nodes. Follow the driver-specific instructions for user/group access.

For a read-only check of the legacy setting:

```sh
grep RmProfilingAdminOnly /proc/driver/nvidia/params
```

On R610+, this flag alone does not account for capability-based grants. Containers also
need the host's profiling access and appropriate device/permission configuration; installing
ncu in a container is insufficient. Follow [NVIDIA's counter-permission guide](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters).
GPUDiagnostics does not change driver permissions or elevate the workload.

### AMD: installation, runtime and performance state

Install ROCprofiler-SDK for the ROCm release used by the workload. Package names vary
by release and distribution; for example, AMD's ROCm 7.1/7.2 repositories provide
`rocprofiler-sdk`. Follow the [installation instructions for that release](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/docs-7.1.1/install/installation.html)
and configure the user's ROCm device access before collecting.

```sh
rocprofv3 --version
rocprofv3 --list-avail
```

The counter list is architecture-specific. Raw counter support does not imply that an
Instinct preset or its derived formulas apply to an RDNA GPU.

On RDNA3/RDNA4, AMD requires the `STABLE_STD` performance level (`profile_standard`
in sysfs) for counters in some hardware blocks. In `AUTO`, wave counts can look valid
while GRBM cycle counters remain zero. Select the target GPU explicitly, record its
original performance level, and restore it after profiling. The package does not change
power settings. See [AMD's PMC profiling prerequisites](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#setting-gpu-performance-level-for-pmc-profiling).

For example, with an amd-smi release supporting these options, replace `<gpu-index>`
with the intended amd-smi GPU index:

```sh
amd-smi list
sudo amd-smi metric --gpu <gpu-index> --perf-level
sudo amd-smi set --gpu <gpu-index> --perf-level STABLE_STD
```

After collection, restore the recorded level (`AUTO` only if it was originally `AUTO`):

```sh
sudo amd-smi set --gpu <gpu-index> --perf-level AUTO
```

Restoring is not housekeeping. `STABLE_STD` fixes the shader clock at a low "standard" level and
disables idle down-clocking: on a W7900 it held 959 MHz against a 2.9 GHz boost, so every other
workload on that GPU ran about three times slower, and idle board power stayed at 94 W instead
of 19 W. It also means profiled durations under `STABLE_STD` are not comparable with unprofiled
runs of the same kernel; compare cycle counters, or time the kernel again in `AUTO`.
When scripting, restore the original state on errors and interrupts as well. GPU indices
can differ between amd-smi, rocprofv3, and DRM sysfs; identify the same device in each tool.
The sysfs knob is unambiguous when the PCI address is known (`card<N>/device/uevent` lists it):

```sh
cat  /sys/class/drm/card<N>/device/power_dpm_force_performance_level          # record: auto
echo profile_standard | sudo tee /sys/class/drm/card<N>/device/power_dpm_force_performance_level
# ... collect ...
echo auto | sudo tee /sys/class/drm/card<N>/device/power_dpm_force_performance_level
```

The HIP/HSA runtime must also support ROCprofiler registration. If the workload runs
but no dispatch records appear, verify that the profiler and workload load compatible,
profiling-enabled runtime libraries. When using an isolated ROCm installation with
AMDGPU.jl, set `ROCM_PATH` as well as the dynamic library search path for that process.

With rocprofv3 1.0.0, an unsupported GPU present alongside the target GPU can cause an
`unordered_map::at` exception during counter configuration. Device-qualified metric
names, such as `metrics = ["SQ_WAVES:device=0", "GRBM_GUI_ACTIVE:device=0"]`, restrict
collection to the selected profiler GPU index. The CSV retains the unqualified names;
use those for `required_metrics`. Enumerating counters or checking that values are finite
does not establish their correctness: validate them against a workload with known counts.

### Verify a small collection first

Run a bounded workload that passes a numerical correctness check, then collect only a
few advertised metrics. For AMD, `SQ_WAVES` and `GRBM_GUI_ACTIVE` provide a useful initial
check when supported; add `:device=N` to each collection metric when needed. For NVIDIA,
start with `gpu__time_duration.sum` and one advertised instruction counter. Validate the
dispatch count, units, and expected activity before collecting a full preset.

The repository's `test/gpu/counter_probe.jl` is a small KernelAbstractions FMA workload
for either vendor, selected with `GPUDIAGNOSTICS_GPU=cuda` or `rocm`. Its three launches
each process 4096 work-items. When checking AMD wave counts, use the kernel's actual
wave width: 4096 work-items correspond to 128 waves at width 32, or 64 at width 64.

| Symptom | First check |
|---|---|
| Collector not found | Installation, `PATH`, or explicit `executable` |
| NVIDIA `ERR_NVGPUCTRPERM` | Driver profiling permissions for the collecting process |
| AMD workload runs but no dispatch CSV appears | HIP/HSA profiler registration, loaded runtime paths, and kernel filter |
| AMD wave counts are plausible but GRBM counts are zero | RDNA performance state before collection |
| rocprofv3 1.0.0 throws `unordered_map::at` with mixed GPUs | Device-qualified metrics for the supported target |
| Metrics absent from otherwise successful output | Target-specific names; pass unqualified CSV names to `required_metrics` |

## Presets and collection behavior

| Intent | AMD (rocprofv3) answers | NVIDIA (ncu) answers |
|---|---|---|
| `:issue` | per-wave issue by class → `amd_insts_per_slot_<class>`, the dynamic mix to hold against the static `kernel_instruction_mix` hot loop | warp and thread instructions → `nvidia_insts_per_slot`; thread / (32 × warp) below 1 is divergence |
| `:occupancy` | resident waves and wait fractions (quad-cycle SQ counters) → latency-bound vs issue-bound | achieved warps per SM over active cycles → `nvidia_active_occupancy` |
| `:memory` | TA / TD busy, TCP stalls, L1 miss → is the in-order vector-memory pipe the wall | DRAM bytes → achieved bandwidth |
| `:fp64` | FMA / ADD / MUL / TRANS F64 per wave → `fp64_flop_per_slot` against the algorithmic count | DFMA / DADD / DMUL per thread → `fp64_flop_per_slot` |
| `:l2` | TCC hits / misses, fabric and HBM reads → does the working set live in L2 or HBM | L2 sector hit rate → `nvidia_l2_sector_hit_rate` |

Each `CounterSet.notes` carries the long form: what the counters are (per-wave, per-thread,
event counts, quad-cycles), the derived keys, the validation and the trap specific to that
preset. Print `COUNTER_SETS[:amd][:memory].notes` before interpreting a collection.

`COUNTER_SETS[vendor][intent]` is a `CounterSet` with native metrics, validation context,
known pass count (or `missing`), and notes. Intents are `:issue`, `:occupancy`, `:memory`,
`:fp64`, and `:l2`. Use `metrics = ["native_metric", ...]` for a custom collection.
AMD presets fit one pass on gfx942 with rocprofv3 1.1.0 / ROCm 7.2.4; parser and derived
quantity tests use saved MI300X captures. Passing a GPU execution test does not validate
hardware-counter collection. Capacity and metric support on other architectures require
validation. NVIDIA metric support and replay cost also vary by GPU and profiler version.
On a bounded sm_120 FMA probe with ncu 2025.4.1,
issue/occupancy/memory used one pass and FP64/L2 used three. These are validation
observations, not portable guarantees; query the target tool first.

Commands use `timeout -k` to bound the profiler and workload if collection hangs. Set
`timeout_s = nothing` when managing process lifetime yourself, including on systems
without that utility.

NVIDIA defaults are `clock_control = :none`, `cache_control = :all`, and
`replay_mode = :kernel`. Profiling can replay kernels, flush caches and serialize work;
its duration is not an ordinary concurrent launch duration. For application-managed
cache priming, consider `replay_mode = :application, cache_control = :none`. This reruns
the application, so ensure repeating its side effects is acceptable. Keep workloads small
and record the settings. See the [NVIDIA profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).

## Reading the counters

The raw values are the profiler's, in the profiler's units. These are the readings that have
already cost time (rocprofv3 1.1.0 on gfx942 unless stated):

1. **Every AMD counter is summed over its hardware dimensions.** `GRBM_GUI_ACTIVE` has
   `DIMENSION_XCC[0:7]` on the MI300X, so a 40 ms dispatch reports 5.7e8 "cycles", a 14 GHz clock,
   until divided by the eight dies. The dispatch's cycle count is `GRBM_GUI_ACTIVE / n_xcd`
   (`Num_Xcc` in the agent info; 1 on single-die parts) and every per-cycle rate uses it. 14 GHz
   means the division was forgotten; "/38" and "/(8 × 304)" are the same per-CU normalisation.
2. **`*_sum` unit counters are event counts summed over instances** (one TA / TD / TCP per CU):
   a unit's busy fraction is `X_BUSY_sum / (cycles × n_cu)`. Never multiply them by the wavefront
   size — doing so once inflated per-slot L1 accesses 64×. Only `SQ_INSTS_*` are per-wave issues,
   and only those take `× wave_size / slots`.
3. **SQ wave-cycle counters are quad-cycles on CDNA and plain cycles on RDNA** (`SQ_WAVE_CYCLES`,
   `SQ_WAIT_ANY`, `SQ_WAIT_INST_ANY`, `SQ_ACTIVE_INST_*`). AMD's own `OccupancyPercent` is
   `400 × SQ_WAVE_CYCLES / max_xcc(GRBM_GUI_ACTIVE) / CU_NUM / 32` on gfx90a, gfx940–942 and gfx950
   and `100 × …` on gfx10, gfx11 and gfx12 (rocprofiler-sdk `counter_defs.yaml`). The parser records
   that unit per dispatch as `device["sq_cycle_unit"]` (4, 1, or `missing` for an architecture outside
   the table; supply it through `device_overrides` there), and resident waves are
   `sq_cycle_unit × SQ_WAVE_CYCLES / cycles`. Ratios between the wave-cycle counters are unit-free.
   `SQ_BUSY_CYCLES` is plain cycles everywhere but is summed over a different set of instances:
   per shader engine on CDNA (`DIMENSION_SHADER_ENGINE`, 32 on the MI300X) and per WGP on RDNA
   (`DIMENSION_WGP × SHADER_ARRAY × SHADER_ENGINE`, 48 on the 96-CU W7900), so `amd_sq_busy` divides
   by `device["sq_instances"]`, resolved the same way (`n_se` on CDNA, `n_cu ÷ 2` on RDNA, or an
   override). Note that AMD normalises by the
   **maximum** over dies while the CSV holds the **sum**, so `/ n_xcd` is the mean over dies: the
   same while every die is busy, below the maximum otherwise.
4. **`amd_active_clock_GHz` is the mean active clock, not the engine clock.** Idle dies count
   no cycles, so it is a lower bound and meaningless for sub-ms dispatches. It agreed with the
   amdgpu sysfs engine clock to 0.01 GHz on the MI300X (1.70 and 1.25 GHz). The MI300X is
   power-managed: the same kernel ran at 1.70 GHz at 7 waves/CU and 1.25 GHz at 14 waves/CU
   against the 750 W board cap, so a launch doing 1.8× the work per cycle finished only 1.27×
   sooner. Compare two runs of the same work in **cycles** (`GRBM_GUI_ACTIVE / n_xcd`,
   `SQ_BUSY_CYCLES`), not seconds, and report the clock next to the timing.
5. **Counter capacity is per hardware block and per pass.** The TCC block takes four counters
   on gfx942; a fifth (`TCC_REQ_sum`, `TCC_READ_sum`) makes rocprofv3 log
   `Request exceeds the capabilities of the hardware to collect`, abort with SIGABRT and leave
   the profiled child hung. `hw_counter_command` wraps the collection in `timeout -k` for exactly
   that failure (the child then exits 137); grep the profiler log for `exceeds the capabilities`
   or `caught signal` when a collection produced no CSV.
6. **AMD profiled durations are close to device-event timings** (26.8–27.4 ms per dispatch
   against a 28.0 ms `LaunchTimer` median): kernels run at speed under `--pmc` and the overhead
   is the counter read. **NVIDIA durations are replays** under `clock_control` / `cache_control`
   and are not benchmarks; instruction counts are exact either way. On the MI300X the FP64 preset
   reproduced the static hot loop's FMA / ADD / MUL / TRANS counts per slot exactly, bit-identical
   across dispatches — the check to repeat on a new architecture before trusting a preset.
7. **NVIDIA thread-instruction metrics are per thread**, so `/ slots` directly, no wave factor.
   `nvidia_fp64_pipe_peak_fraction` needs `sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active`,
   which is not in the `:fp64` preset; add it through `metrics` when the question is "is the FP64
   pipe the wall" (a kernel at 88 % pipe fraction gained only 7–9 % from removing integer work).
8. **gfx1100 (Radeon Pro W7900) is validated for the SQ counters, nothing else.** It exposes no
   F64, TA / TD / TCP or TCC counters, so only `SQ_*` and `GRBM_GUI_ACTIVE` can be collected and
   the `:memory`, `:fp64` and `:l2` presets do not apply. A W7900 collection (fixture `w7900_*`)
   confirmed the RDNA row of the table: `SQ_INSTS_VALU × 32 / slots` is 1.0001 per FMA, resident
   waves come out at 1346 of the 3072 the hardware holds (a ×4 unit would exceed it), and
   `SQ_BUSY_CYCLES` is 0.89 of the dispatch per WGP (7.1 per shader engine). Three things to know:
   in `AUTO` the run "succeeds" with `SQ_WAVES` correct while `GRBM_GUI_ACTIVE` and
   `SQ_WAVE_CYCLES` read zero, and `STABLE_STD` pins the shader clock at 959 MHz, so profiled
   durations are three times an unprofiled run; with the host's integrated GPU present,
   `ROCR_VISIBLE_DEVICES` does not hide it from rocprofv3 1.1 and unqualified counters abort with
   `unordered_map::at` — device-qualified names (`SQ_WAVES:device=0`) are mandatory; and
   `SQ_WAVES` read 14× too high on one dispatch in three of five collections, so validate the
   wave count per dispatch rather than trusting a median. `amd_active_clock_GHz` read 1.03 GHz
   against the 959 MHz sysfs shader clock, consistent with RDNA3's front-end clock running above
   the shader clock; treat it as the front-end clock there. The container recipe that ran:
   `rocm/dev-ubuntu-24.04` plus `libdw1`, `--device=/dev/kfd --device=/dev/dri
   --security-opt seccomp=unconfined --ipc=host`, the host's juliaup and depot bind-mounted with
   `HOME` and `JULIA_DEPOT_PATH` set, and `ROCM_PATH=/opt/rocm`.

Kernel names as the tools report them: KernelAbstractions kernels are `gpu_<kernel name>(…)`,
AcceleratedKernels' `foreachindex` is `gpu__forindices_global_(…)`, and runtime-internal
`__amd_rocclr_*` dispatches are excluded by the default kernel selection.

## Portable input and retained data

`hw_counters` accepts a directory or CSV and detects the tool from its header. Multiple
collections are an error: provide `name`, `vendor`, or the CSV path. Kernel selection must
resolve to one distinct full name. Each selected dispatch retains its own device, process,
context, queue, launch geometry and resources. Tool IDs keep their original indexing;
dispatch ID alone need not be globally unique. AMD agent metadata is matched by ID.

NVIDIA wide raw exports (including their units row) and long metric/value exports are
supported. Quoted signatures and numeric grouping separators are accepted. Unknown counters
are `missing`, not zero. Conflicting duplicate counters, units or identities are errors.
Categorical NVIDIA device/launch attributes stay in dispatch metadata. Native numeric
values retain their units; the command builder requests base units.

Native access is `hc["SQ_WAVES"]` or `hc["smsp__inst_executed.sum"]`. `hc.units` records
ncu's units; AMD CSV does not supply units, so those entries are `missing`. `slots` can
be one positive count or a vector aligned with the selected launches. Supply missing
known device properties explicitly:

```julia
hc = hw_counters(:amd, "prof"; name = "cell", kernel = "my_kernel",
    device_overrides = Dict(1 => Dict("architecture" => "gfx942", "n_xcd" => 8,
        "n_cu" => 304, "wave_size" => 64, "max_waves_per_cu" => 32)))
```

Cycle normalisation applies wherever `n_xcd` is known (agent info or override); the only
architecture-specific input is how the SQ block reports, taken from AMD's own definitions per
family or from `device_overrides`: `"sq_cycle_unit"` (4 on CDNA, 1 on RDNA) and `"sq_instances"`
(shader engines on CDNA, WGPs on RDNA). Fixture-validated on gfx942; the RDNA entries follow AMD's
`counter_defs.yaml` and a W7900 collection. Use actual known device properties for overrides.

Binary `.ncu-rep` input requires an explicit export using the system tool:

```julia
run(hw_counter_export_command("cell.ncu-rep"; output = "cell_ncu.csv"))
hc = hw_counters("cell_ncu.csv")
```

Provenance accepts only `tool_version`, `clock_control`, `cache_control`, `replay_mode`,
`counter_set`, and `passes`. Supply known settings; CSV need not contain them. Parsers
discard host names, executable names and tool log messages. Raw reports can contain
machine information; sanitize them before publishing fixtures.

## Derived quantities

`hw_counter_derived` returns columns, not medians. Missing inputs and zero denominators
give `missing`; absent metric families are omitted. Summaries skip missing samples and
report the number of contributing observations.

| Key | Meaning |
|---|---|
| `insts_per_slot_fp64_fma/add/mul` | NVIDIA predicated-on thread instructions / slots; AMD wave instructions × wave size / slots (assumes full active waves; divergence can overestimate active-lane work). |
| `fp64_flop_per_slot` | Twice FMA plus add and multiply counts; excludes transcendental seeds and matrix work. |
| `amd_active_clock_GHz` | `GRBM_GUI_ACTIVE / n_xcd / duration_s / 1e9`; equals engine clock only while every die stays busy. |
| `amd_elapsed_occupancy` | `sq_cycle_unit × SQ_WAVE_CYCLES / (GRBM_GUI_ACTIVE / n_xcd) / n_cu / max_waves_per_cu`; the unit is 4 on CDNA, 1 on RDNA. |
| `nvidia_active_occupancy` | SM active-warp percentage of sustained peak over active cycles / 100. Its averaging window differs from AMD's estimate. |
| `amd_l2_request_hit_rate` | TCC hits / (hits + misses). |
| `nvidia_l2_sector_hit_rate` | L2 sector lookup hit percentage / 100; sectors and requests are distinct denominators. |
| `nvidia_fp64_pipe_peak_fraction` | FP64 pipe percentage of sustained peak over active cycles / 100; not a portable fraction of wall time issuing FP64. |

AMD issue, wait, residency and unit-busy estimates keep `amd_` names. The formulas retain
the SQ wave-cycle unit and the die / engine dimensions AMD's own derived metrics use. Native
counts remain available to audit them. Profiler local-memory traffic can include call-ABI stack frames
as well as actual register spills.

## Migration from 0.2

Version 0.3 removes the AMD-only public family:

| Old | Replacement |
|---|---|
| `rocprof_command(cmd; counters = :sq_issue, ...)` | `hw_counter_command(ROCBackend(), cmd; set = :issue, ...)`, or `RocprofV3()` without AMDGPU |
| `ROCPROF_COUNTER_SETS` | `COUNTER_SETS[:amd]` (`:sq_waves` → `:occupancy`, `:l1_pipe` → `:memory`) |
| `rocprof_available()` | `hw_counters_available(ROCBackend())`, or `hw_counters_available(RocprofV3())` without AMDGPU |
| `rocprof_counters(dir; ...)` | `hw_counters(:amd, dir; ...)` |
| `RocprofCounters` | `HWCounters`, with metadata per `HWDispatch` |
| `rocprof_derived(rc)` | `hw_counter_derived(hc)` returns **vectors**; summarize explicitly |
| `rocprof_median(rc, counter)` | `Statistics.median(skipmissing(hc[counter]))` when samples exist |
| `rocprof_summary(rc)` | `hw_counter_summary(hc)` |
| `fp64_flop_per_slot` = 2·FMA + ADD + MUL + TRANS | `fp64_flop_per_slot` = 2·FMA + ADD + MUL on both vendors; the AMD TRANS count stays in `amd_insts_per_slot_valu_trans_f64` |

Schema 2 replaces counter summary keys with `raw_<name>_median` and
`derived_<name>_median`, plus ranges and valid sample counts. Other result layouts are
unchanged. Changing a prefix alone does not migrate old reports. Unknown device properties
are missing; summaries include only properties common to all selected dispatches.
The result holds selected counter dispatches; an AMD kernel-trace companion is no longer
folded into the counter result.
