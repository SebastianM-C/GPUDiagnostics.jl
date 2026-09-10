```@meta
CurrentModule = GPUDiagnostics
```

# Porting a backend

An extension declares its features with `GPUDiagnostics.supports(::MyBackend, ::Val{:feature}) = true`
and implements the hooks behind them. `:devices` / `:device_props` / `:telemetry` are the `gpu_*`
generics of the device API plus `gpu_sampler_sources` (and a `sampler_source(::Val{kind}, …)` for a new
source kind); `:events` is `gpu_event` / `gpu_elapsed`; `:kernel_inventory` is `backend_compiled_kernels`
(with `backend_wrap_kernel` if the kernel type is shaped differently); `:resources` / `:occupancy` are
`backend_kernel_attributes` / `backend_kernel_occupancy` (+ optional `backend_kernel_isa_info`);
`:native_mix` is `backend_kernel_machine_code`; `:ir_mix` is `backend_kernel_ir_counts`, the one hook
every GPU backend should have (a GPUCompiler job over LLVM IR). The `backend_*` names are public API,
documented in their docstrings, not exported.

## The hooks

```@docs
backend_compiled_kernels
backend_wrap_kernel
backend_kernel_attributes
backend_kernel_occupancy
backend_kernel_isa_info
backend_kernel_machine_code
backend_kernel_ir_counts
backend_versions
```

The telemetry hooks `gpu_sampler_sources` and `sampler_source` are documented in the [API reference](api.md).

## What Metal (or any backend without FP64, a public ISA or telemetry) declares

`:devices`, `:events`, `:kernel_inventory`, `:ir_mix`, `:peak_flops` (Float32 only, so not
`:fp64`), a partial `:resources` (no register count, `occupancy = missing`). Everything else stays
undeclared and the conformance suite checks that it fails cleanly. A backend that supplies only
the IR mix gets a coarser but honest instruction picture — backend FMA contraction is invisible
at IR level, which the caveats page explains.
