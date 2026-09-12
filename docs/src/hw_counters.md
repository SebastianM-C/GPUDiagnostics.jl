# Hardware counters

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

When scripting, restore the original state on errors and interrupts as well. GPU indices
can differ between amd-smi, rocprofv3, and DRM sysfs; identify the same device in each tool.

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

Architecture-specific AMD normalization is currently validated for gfx942. Other targets
retain raw counters and dimension-free ratios; unvalidated cycle/occupancy estimates are
`missing`. Use actual known device properties for overrides.

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
| `amd_elapsed_occupancy` | On gfx942: `4 × SQ_WAVE_CYCLES / (GRBM_GUI_ACTIVE / n_xcd) / n_cu / max_waves_per_cu`. |
| `nvidia_active_occupancy` | SM active-warp percentage of sustained peak over active cycles / 100. Its averaging window differs from AMD's estimate. |
| `amd_l2_request_hit_rate` | TCC hits / (hits + misses). |
| `nvidia_l2_sector_hit_rate` | L2 sector lookup hit percentage / 100; sectors and requests are distinct denominators. |
| `nvidia_fp64_pipe_peak_fraction` | FP64 pipe percentage of sustained peak over active cycles / 100; not a portable fraction of wall time issuing FP64. |

AMD issue, wait, residency and unit-busy estimates keep `amd_` names. The gfx942 formulas
retain quad-cycle SQ normalization and die/engine dimensions. Native counts remain
available to audit them. Profiler local-memory traffic can include call-ABI stack frames
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

Schema 2 replaces counter summary keys with `raw_<name>_median` and
`derived_<name>_median`, plus ranges and valid sample counts. Other result layouts are
unchanged. Changing a prefix alone does not migrate old reports. Unknown device properties
are missing; summaries include only properties common to all selected dispatches.
The result holds selected counter dispatches; an AMD kernel-trace companion is no longer
folded into the counter result.
