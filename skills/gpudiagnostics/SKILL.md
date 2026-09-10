---
name: gpudiagnostics
description: Use GPUDiagnostics.jl to answer "why is this KernelAbstractions kernel slow / what did the GPU do / what will it do on a GPU I don't have" — which instrument answers which question, the rules an agent gets wrong without being told (capabilities, missing vs nothing, what the classes mean, busy medians), how to store results and how to verify a change on hardware. Use when profiling or diagnosing a KA kernel on NVIDIA or AMD, reading a GPUDiagnostics manifest section, or changing GPUDiagnostics itself. Pointers into the docs, not a copy of them.
---

# GPUDiagnostics.jl — decision guide

**Read the docs as files, not URLs.** The documentation sources ship inside the installed package, so
read them verbatim instead of fetching the site (a URL fetch returns a lossy summary):

```
julia -e 'using GPUDiagnostics; println(pkgdir(GPUDiagnostics))'   # then read <pkgdir>/docs/src/<page>.md
```

Pages (`docs/src/`): `capabilities.md`, `device_timing.md`, `telemetry.md`, `peaks_probes.md`,
`resources_mix.md`, `hw_counters.md`, `report.md`, `porting.md`, `caveats.md`; the docstrings are in
`src/*.jl` and the hardware suite in `test/gpu/runtests.jl`. The rendered site
(https://sebastianm-c.github.io/GPUDiagnostics.jl/dev/) is for humans. Everything dispatches on the
KernelAbstractions `Backend`; load CUDA.jl or AMDGPU.jl for the vendor methods.

## Symptom → instrument

| you see | reach for | read (`docs/src/`) |
|---|---|---|
| Many small launches; wall time ≫ kernel time | `measure_launch_overhead(backend)` for the per-launch floor; `LaunchTimer` medians vs `first_launch_s` | `peaks_probes.md`, `device_timing.md` |
| Kernel slower than its FLOP count predicts | `fp64_issue_floor(mix; n_slots, peak_fp64_flops = measure_peak_flops(backend), kernel_time_s)` — the fraction of the launch the FP64 pipe must be issuing; then GPM `fp64_util` from the sampler on NVIDIA | `resources_mix.md`, `telemetry.md` |
| Adding work stops scaling | `kernel_resources` (theoretical occupancy) against sampler `sm_occupancy_busy_mean` (achieved); `power_capped_fraction` and `sm_clock_MHz_busy_median` — a power-bound kernel gains per cycle, not per second | `resources_mix.md`, `telemetry.md` |
| Results differ bitwise across vendors or compile options | `kernel_ir_mix` `fp64_contract` vs the native `fp64_fma` count: backend FMA contraction | `resources_mix.md`, `caveats.md` |
| "Will it fit / how will it run on the MI300X / H100 I have not rented" | `kernel_instruction_mix(backend, ck; target = "gfx942")` / `"sm_90"` cross-compiles the same kernel from any device of the vendor | `resources_mix.md` |
| Run was slow on a rented box, GPU numbers look fine | `host_snapshot(backend)`: Julia threads vs the cgroup CPU quota, BLAS threads, driver / runtime versions; `warnings` is data | `peaks_probes.md` |
| Need per-dispatch hardware counters on AMD | `rocprof_command` wraps the process, `rocprof_counters(dir)` parses, `rocprof_derived` normalises (cycles = `GRBM_GUI_ACTIVE / n_xcd`) | `hw_counters.md` |
| Need the results in a manifest / table | `diagnostics_dict(x; prefix)` per result; `GPUTelemetry` is a Tables.jl table | `report.md` |

## Rules an agent gets wrong without being told

- **Check before calling.** `supports(backend, :telemetry)` / `capabilities(backend)`; an undeclared
  feature throws `BackendUnsupported` (its message names the package to load). Do not `try`/`catch`
  around every call — branch on the capability.
- **`missing` ≠ `nothing`.** `missing` = the vendor cannot report it (HIP const size, SASS register
  count, occupancy without a calculator); `nothing` = does not exist / not asked for (a kernel with no
  loop → `hot_loop === nothing`, `ir = false`). Never turn either into `-1` or `NaN` in stored data:
  `diagnostics_dict` omits `missing` keys, and every dict carries `gpudiagnostics_schema`.
- **Static counts are code, not execution.** `kernel_instruction_mix(...).total` includes cold
  exception paths; `.hot_loop.counts` is the per-iteration floor, nested loops counted once.
- **`atomic_fallback` is not spill traffic.** It is the private/shared address-space path LLVM emits
  for an atomic on a generic pointer; an address-space-1 pointer removes it. `local_mem_bytes` /
  `scratch_bytes` from `kernel_resources` is the spill figure.
- **`compute_util` is per-vendor.** NVML = fraction of time any kernel was resident; amdgpu = closer to
  SM busy. Compare within a vendor only. `thread_fill_occupancy` is a static upper bound.
- **Read busy medians.** `<col>_busy_mean` / `_busy_median` (rows with `compute_util ≥ 0.5`), not window
  means; the sampler child needs a few seconds to start on NVIDIA (`first_sample_s`), so a window
  shorter than that has no samples. Separate the JIT launch: `launch_times(timer; skip_first = true)`.
- **Device order and count.** CUDA numbers devices fastest-first (not by PCI slot); AMDGPU.jl on an APU
  host exposes the integrated GPU as a device. Check `gpu_name(backend)` after `gpu_device!`.
- **Peak probe.** `measure_peak_flops(backend, T)` is the FMA-chain rate at the clocks the device holds
  (≈ 91 % of spec on an H100, ≈ 54 % on an A100 — a known probe-geometry question), never a matrix-unit
  peak; `Float64` requires the `:fp64` capability.

## Storing results (manifest convention)

`merge!(section, diagnostics_dict(kernel_resources(b, ck); prefix = "kernel_"),
diagnostics_dict(mix; prefix = "kernel_mix_"), diagnostics_dict(mix.ir; prefix = "kernel_ir_"),
diagnostics_dict(timer; prefix = "kernel_"), diagnostics_dict(telem; prefix = "sampler_"),
diagnostics_dict(host_snapshot(b)))`. Keys are additive and never renamed; the IR counts are a
separate dict from the mix on purpose. Print any result for a human: `show(stdout, MIME"text/plain"(), x)`.

## Changing GPUDiagnostics itself

1. Unit suite without a GPU: `julia --project -e 'using Pkg; Pkg.test()'` (Aqua included — a duplicated
   struct or method overwrite fails precompile there, not in the REPL).
2. Hardware suite, one vendor per process, `Pkg.resolve()` first (the env dev-tracks the package):
   `GPUDIAGNOSTICS_GPU=cuda|rocm julia --project=test/gpu -e 'using Pkg; Pkg.resolve(); Pkg.instantiate(); include("test/gpu/runtests.jl")'`.
   With two visible devices the multi-device section runs.
3. A new capability = a `supports` method per extension + a conformance-suite branch (declared: shapes;
   undeclared: `BackendUnsupported`). A new hook = a `backend_*` name with a docstring on the porting
   page; private helpers keep the underscore and must not be `@ref`'d.
4. A value the vendor may not report is `missing` at the API boundary; `NaN` only inside the telemetry
   matrix; `nothing` only for "does not exist".
5. No hostnames or other infrastructure details in commits, PRs or issues; hardware model names are fine.
