```@meta
CurrentModule = GPUDiagnostics
```

# GPUDiagnostics.jl

Vendor-neutral runtime diagnostics for [KernelAbstractions](https://github.com/JuliaGPU/KernelAbstractions.jl)
code. Everything dispatches on the KA `Backend`; CUDA.jl and AMDGPU.jl package extensions supply the
vendor methods, and the `CPU` backend gets host fallbacks so the plumbing runs (and is tested) without a GPU.
No kernel is modified by any of it.

Not registered yet: `pkg> add https://github.com/SebastianM-C/GPUDiagnostics.jl`.

```julia
using GPUDiagnostics, CUDA          # or AMDGPU
backend = CUDABackend()

gpu_device_count(backend), gpu_name(backend), gpu_arch(backend)
gpu_sm_count(backend) * gpu_max_threads_per_sm(backend)   # resident-thread capacity
gpu_memory_info(backend), gpu_power(backend), gpu_utilization(backend)
```


## What is where

- [Capabilities](capabilities.md) — which subsystems a backend implements, `BackendUnsupported`, the conformance suite.
- [Device API and kernel timing](device_timing.md) — device enumeration and properties, device-event `LaunchTimer`.
- [Telemetry](telemetry.md) — the out-of-process sampler, its columns, GPM counters, stats.
- [Measured peak and cheap probes](peaks_probes.md) — FMA-chain peak FLOP/s, launch overhead, host snapshot, warm-up accounting.
- [Resource report and instruction mix](resources_mix.md) — registers / spills / occupancy, the static ISA and IR mix, cross-compilation, the FP64 issue floor.
- [Hardware counters](hw_counters.md) — rocprofv3 / Nsight Compute collectors and a shared per-dispatch result.
- [Report layer](report.md) — `diagnostics_dict`, `show`, Tables.jl.
- [Porting a backend](porting.md) — the `backend_*` contract.
- [Caveats](caveats.md) — the traps that have already cost time.
- [API reference](api.md)
