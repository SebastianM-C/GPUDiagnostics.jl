# Capabilities

```julia
capabilities(CUDABackend())        # [:devices, :device_props, :events, :telemetry, …]
supports(backend, :native_mix)     # Bool; the CPU backend has :devices, :events, :peak_flops, :fp64, :kernel_inventory
```

A value a backend cannot answer is `missing` (HIP has no const-size attribute, SASS lists no register count,
a runtime without an occupancy calculator has no `occupancy`); `NaN` appears only inside the telemetry
matrix, where rows must stay numeric; `nothing` means "does not exist / not asked for" (a kernel with no
loop, `ir = false`). Every entry point belongs to one feature of `FEATURES`; calling one the backend does not declare throws
`BackendUnsupported`, whose message names the package to load when that is the reason. The conformance
suite (`test/conformance.jl`) checks a backend against its declarations and runs on the CPU backend in
the regular tests.



## Conformance suite

`test/conformance.jl` checks a backend against its declarations: every declared feature's entry
points return the documented shapes, every undeclared one throws `BackendUnsupported` — never a
`MethodError`, never a vendor-named string. It runs on the CPU backend in the regular tests and,
with a compiled kernel, on real hardware through the hand-run `test/gpu` suite:

```
GPUDIAGNOSTICS_GPU=cuda julia --project=test/gpu -e 'using Pkg; Pkg.resolve(); Pkg.instantiate(); include("test/gpu/runtests.jl")'
GPUDIAGNOSTICS_GPU=rocm …
```
