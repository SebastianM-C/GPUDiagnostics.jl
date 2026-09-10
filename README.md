# GPUDiagnostics.jl

[![CI](https://github.com/SebastianM-C/GPUDiagnostics.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/SebastianM-C/GPUDiagnostics.jl/actions/workflows/CI.yml)
[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://SebastianM-C.github.io/GPUDiagnostics.jl/)

Vendor-neutral runtime diagnostics for [KernelAbstractions](https://github.com/JuliaGPU/KernelAbstractions.jl)
kernels. Everything dispatches on the KA `Backend`; the CUDA.jl and AMDGPU.jl extensions supply the
vendor methods, and the `CPU` backend gets host fallbacks so the plumbing runs (and is tested) without a
GPU. No kernel is modified by any of it.

Not registered yet: `pkg> add https://github.com/SebastianM-C/GPUDiagnostics.jl`.

## What it answers

| question | instrument | docs |
|---|---|---|
| Which device, how many, how much memory, power, utilization? | `gpu_device_count`, `gpu_device!`, `gpu_memory_info`, `gpu_power`, … | [device API](https://SebastianM-C.github.io/GPUDiagnostics.jl/device_timing/) |
| How long does each kernel launch take, when launches are queued asynchronously? | `LaunchTimer` (one device-event pair per launch), `first_launch_s` | [kernel timing](https://SebastianM-C.github.io/GPUDiagnostics.jl/device_timing/) |
| What did the device do while the code ran — power, clocks, temperature, achieved occupancy? | `with_gpu_sampler` (an out-of-process child, one TSV row per device per tick), `gpu_telemetry_stats` | [telemetry](https://SebastianM-C.github.io/GPUDiagnostics.jl/telemetry/) |
| What FP64 / FP32 rate can scalar code reach on this device, at the clocks it holds? | `measure_peak_flops` (FMA-chain probe, never routed to matrix units) | [measured peak](https://SebastianM-C.github.io/GPUDiagnostics.jl/peaks_probes/) |
| What does a launch cost; what is this host, really? | `measure_launch_overhead`, `host_snapshot` (threads vs cgroup quota, driver / runtime versions) | [probes](https://SebastianM-C.github.io/GPUDiagnostics.jl/peaks_probes/) |
| Registers, spills, shared memory, theoretical occupancy of the kernels that actually ran? | `compiled_kernels`, `kernel_resources` → `KernelResources` | [resource report](https://SebastianM-C.github.io/GPUDiagnostics.jl/resources_mix/) |
| What instructions is the kernel made of — here, or on a GPU I have not rented yet? | `kernel_instruction_mix` (AMD ISA / NVIDIA SASS by class, per loop; `target = "gfx942"` cross-compiles), `kernel_ir_mix`, `fp64_issue_floor` | [instruction mix](https://SebastianM-C.github.io/GPUDiagnostics.jl/resources_mix/) |
| Hardware counters per dispatch on AMD? | `rocprof_command`, `rocprof_counters`, `rocprof_derived` (rocprofv3 wrapper + parser) | [hardware counters](https://SebastianM-C.github.io/GPUDiagnostics.jl/hw_counters/) |
| How do I store or read all of that? | `diagnostics_dict(x; prefix)` (flat TOML-safe dicts, stable keys), `show`, Tables.jl on `GPUTelemetry` | [report layer](https://SebastianM-C.github.io/GPUDiagnostics.jl/report/) |

Which of these a backend implements is declared through `supports(backend, :feature)` /
`capabilities(backend)`; an entry point of an undeclared feature throws `BackendUnsupported`. A value a
backend cannot report is `missing`, never a sentinel.

## Thirty seconds

```julia
using GPUDiagnostics, CUDA                      # or AMDGPU
backend = CUDABackend()

timer = LaunchTimer()
result, telem = with_gpu_sampler(backend, 0.5) do
    lane = launch_lane(timer, backend)
    for item in work
        e0 = launch_tick(timer, backend)
        my_kernel!(backend)(item; ndrange = n)  # asynchronous
        launch_tock!(timer, lane, backend, e0)
    end
    KernelAbstractions.synchronize(backend)
end

timer                                           # per-device first / median / max launch seconds
telem                                           # ticks, window, power / clock / occupancy means and busy medians
ck = only(compiled_kernels(backend; pattern = r"my_kernel"))
kernel_resources(backend, ck)                   # registers, spills, LDS, theoretical occupancy
m = kernel_instruction_mix(backend, ck)         # class × loop table; m.hot_loop is the per-slot floor
fp64_issue_floor(m; n_slots, peak_fp64_flops = measure_peak_flops(backend), kernel_time_s = median_launch_s)

merge!(manifest, diagnostics_dict(timer; prefix = "kernel_"), diagnostics_dict(telem; prefix = "sampler_"))
```

## Reading the numbers

The [caveats page](https://SebastianM-C.github.io/GPUDiagnostics.jl/caveats/) collects the traps
that have already cost time: static counts are code, not execution; `compute_util` is not comparable
across vendors; `thread_fill_occupancy` is an upper bound; the `atomic_fallback` class is not spill
traffic; read busy medians, not window means. The
[`skills/gpudiagnostics`](skills/gpudiagnostics/SKILL.md) file is the same material as a one-page
decision guide for coding agents.

## Testing

`Pkg.test()` runs without a GPU (CPU fallbacks, fixtures of real rocprofv3 collections, a conformance
suite over the capability declarations). The hand-run hardware suite exercises the vendor paths:

```
GPUDIAGNOSTICS_GPU=cuda julia --project=test/gpu -e 'using Pkg; Pkg.resolve(); Pkg.instantiate(); include("test/gpu/runtests.jl")'
GPUDIAGNOSTICS_GPU=rocm …
```

It has been run on RTX 4080 SUPER, A100, H100 (also both in one process) and Radeon Pro W7900.

## Porting a backend

A backend declares its features with `GPUDiagnostics.supports(::MyBackend, ::Val{:feature}) = true`
and implements the `backend_*` hooks behind them. The
[porting page](https://SebastianM-C.github.io/GPUDiagnostics.jl/porting/) lists them; `:ir_mix` (a
GPUCompiler job over LLVM IR) is the one every GPU backend should have.

## Related packages

- [GPUInspector.jl](https://github.com/pc2/GPUInspector.jl) — NVIDIA-only inspection and micro-benchmarks
  (peak FLOPS, memory bandwidth, host-device / peer-to-peer transfers, stress tests). GPUDiagnostics is
  vendor-neutral through the KernelAbstractions backend, and adds the compile-time / static-code side
  (resource report, instruction mix, cross-compilation) plus the AMD rocprofv3 wrapper; it does not
  duplicate the bandwidth benchmarks.
- [NVTX.jl](https://github.com/JuliaGPU/NVTX.jl) — range and mark annotations for Nsight Systems
  timelines. Complementary: NVTX labels host regions for a profiler GUI, `LaunchTimer` measures
  individual kernels with device events at run time, without a profiler attached.
- [LIKWID.jl](https://github.com/JuliaPerf/LIKWID.jl) — hardware performance counters through LIKWID,
  primarily for the CPU (LIKWID also has NVIDIA and ROCm counter backends). GPUDiagnostics reaches the
  GPU counters through the vendor tools instead.
- CUDA.jl's `@profile` and AMDGPU.jl's profiling — the vendor packages' own integrated profilers, for
  per-kernel timelines of a single session.

## Provenance

Written with the help of [Claude Code](https://claude.com/claude-code) and developed and tested against
the needs of [ElectronDynamicsModels.jl](https://github.com/SebastianM-C/ElectronDynamicsModels.jl),
where it started as an in-repo sub-package. Design decisions, validation on real hardware and the
measurements quoted in the docs are the maintainer's; expect rough edges outside the paths that project
exercises.
