# Caveats

Things that have already cost time, collected so they are read rather than remembered.

## Static counts are code, not execution

`kernel_instruction_mix` counts every instruction of the binary once, cold exception paths
included (bounds checks, `DomainError` helpers dominate a Julia kernel's code size and never
execute). The dynamic figure it stands in for is the hot loop's count: read `hot_loop` for the
per-iteration floor and `total` for code size. Nested loops are counted once inside their parent,
so an inner Newton loop of `n` iterations adds `(n − 1)` × its own count per slot on top.

## The instruction-mix `atomic_fallback` class

An atomic on a generic (flat) pointer makes LLVM emit private- and shared-address fallback paths
(`scratch_load/store`, `ds_*`) that a device-memory atomic never executes. They are counted as
`atomic_fallback`, not as memory traffic — read them as "these atomics take a generic pointer",
not as spills. An address-space-1 pointer removes the paths.

## IR mix versus machine mix

`kernel_ir_mix` counts the optimized LLVM IR *before* the backend: an IR `fma` count of 0 against
hundreds of `v_fma_f64` / DFMA is the backend's FMA contraction (Julia lowers `muladd` to a
`contract`-flagged `fmul`/`fadd` pair, counted as `fp64_contract`); IR `div` / `sqrt` against the
machine's reciprocal seeds + FMA sequences is its expansion. A backend that offers only the IR mix
therefore reports the arithmetic as written, not as issued.

## `fp64_util` is normalised to full-rate issue slots

The GPM `fp64_util` column is the fraction of the SM's full-rate FP64 issue slots: a saturated
FP64 FMA chain reads ≈ 0.9 on an H100 but only ≈ 1.5 % on a 1/64-rate consumer board.

## `compute_util` means different things per vendor

NVML's utilization is the fraction of the sample period during which *any* kernel was resident,
not SM busy; the amdgpu driver's `gpu_busy_percent` is closer to SM busy. Do not average
`compute_util` across vendors; compare within one.

## `thread_fill_occupancy` is an upper bound

It is the static ratio of a launch's threads to the device's resident-thread capacity. A memory-
or power-bound kernel does not get faster by filling more of it (an MI300X split-mode experiment
fit two groups at 90 % fill and gained 1.16×, power-pinned at TDP throughout). Hold it against the
sampler's achieved occupancy and clocks.

## `shared_mem_bytes` is what the compiler reserved

LLVM's AMDGPU backend promotes private arrays it cannot keep in registers to LDS, sized for the
kernel's maximum block size. `kernel_resources` reports that reservation — not what the source
declared — because the reservation is what caps resident blocks per CU.

## Nsight Compute's "local memory spilling" counts the call ABI too

On NVIDIA, out-of-line device functions round-trip sret/byval frames through local memory; ncu
labels that as spilling. `kernel_resources` separates the call-ABI stack frame from true register
spills via `ptxas --verbose`, and `CUDABackend(always_inline = true)` removes the out-of-line
functions altogether (a 2.48× win on one production kernel).

## The AMD ISA dump regenerates with default compile options

`kernel_resources`' AMD ISA figures come from a `code_native` of the kernel's job with default
options; a non-default production option is not reflected there.

## Telemetry means over a window that includes JIT and idle are not the kernel's numbers

Read the `_busy_mean` / `_busy_median` statistics (rows with `compute_util ≥ 0.5`), and separate
the warm phase with `first_launch_s` / `launch_times(timer; skip_first = true)`. The sampler child
takes a few seconds to start on NVIDIA (it loads CUDA.jl for NVML); `first_sample_s` records it and
a window shorter than that has no samples.

## The launch probe's device time includes the event pair

`measure_launch_overhead`'s `device_s` brackets a one-instruction kernel with two device events,
whose own cost is a few microseconds — it is an upper bound and may exceed the host round trip.

## The peak probe reads low on some architectures

The FMA-chain probe reaches ≈ 91 % of the vector FP64 spec on an H100 but ≈ 54 % on an A100 at
full boost clock; the shortfall is a probe-geometry question (chains per thread, launch size)
tracked in the repository's issues, not a clock effect. Treat the number as "attainable by scalar
code at the clocks the device holds", and cross-check with the sampler's clock columns.

## `test/gpu` needs `Pkg.resolve()` first

The `test/gpu` environment dev-tracks the package; a manifest resolved before a dependency was
added to `Project.toml` is not refreshed by `instantiate` alone.
