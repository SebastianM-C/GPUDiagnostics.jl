# Report layer

```julia
merge!(manifest["gpu"], diagnostics_dict(kernel_resources(backend, ck); prefix = "kernel_"),
                        diagnostics_dict(m; prefix = "kernel_mix_"), diagnostics_dict(m.ir; prefix = "kernel_ir_"),
                        diagnostics_dict(timer; prefix = "kernel_"), diagnostics_dict(telem; prefix = "sampler_"))
r                                   # every result type prints as a small table (show)
using DataFrames; DataFrame(telem)  # GPUTelemetry is a Tables.jl column table
```

`diagnostics_dict` gives a flat `Dict{String, Any}` of TOML-safe scalars under stable keys: keys are
additive and never renamed, a value the backend could not report is omitted rather than written as a
sentinel, and every dict carries `gpudiagnostics_schema`. The IR counts are a separate dict so the
`kernel_mix_` and `kernel_ir_` families stay apart.

## Related packages

- [GPUInspector.jl](https://github.com/pc2/GPUInspector.jl) — NVIDIA-only inspection and micro-benchmarks
  (peak FLOPS, memory bandwidth, host-device / peer-to-peer transfers, stress tests). GPUDiagnostics is
  vendor-neutral through the KernelAbstractions backend, and adds the compile-time / static-code side
  (resource report, instruction mix, cross-compilation) plus the AMD rocprofv3 wrapper.
- [NVTX.jl](https://github.com/JuliaGPU/NVTX.jl) — range and mark annotations for Nsight Systems
  timelines. Complementary: NVTX labels host regions for a profiler GUI, `LaunchTimer` measures
  individual kernels with device events at run time, without a profiler attached.
- [LIKWID.jl](https://github.com/JuliaPerf/LIKWID.jl) — hardware performance counters through
  LIKWID, primarily for the CPU; LIKWID also has NVIDIA (`nvmon`) and ROCm (`rocmon`) counter
  backends. GPUDiagnostics reaches the GPU counters through the vendor tools instead (NVML GPM
  counters in-process on NVIDIA, rocprofv3 as a wrapping process on AMD).
- [CUDA.jl's `@profile`](https://cuda.juliagpu.org/stable/development/profiling/) and
  [AMDGPU.jl's `@roc` profiling](https://amdgpu.juliagpu.org/stable/profiling/) — the vendor packages'
  own integrated profilers, for per-kernel timelines of a single session.
