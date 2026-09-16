# Measured peak and cheap probes

```julia
peak_flops_probe(backend)              # PeakProbe: FP64 FLOP/s + the geometry that attained it + the sweep
measure_peak_flops(backend)            # the FLOP/s alone (peak_flops_probe(...).flops)
measure_peak_flops(backend, Float32)   # the FP32 rate of the same probe
measure_gemm_flops(backend)            # the matrix-unit (BLAS) rate: an UPPER reference, not the peak
diagnostics_dict(peak_flops_probe(backend); prefix = "peak_probe_")   # for a run manifest
```

No per-architecture table: the probe measures the attainable vector rate at the clocks the device
actually holds, and is never routed to matrix/tensor units (the wrong yardstick for scalar kernels).
The result is checked against a host reference so a mis-launched kernel cannot be credited. It runs on
the CPU backend too, one scalar chain per work-item, which under-reports the host by its SIMD width;
the vectorised host number is `LinearAlgebra.peakflops(2048; ntrials = 3)`, a different quantity.

## Geometry

How much of the FP64 pipe a dependent-FMA chain can fill depends on the chains per thread (the
instruction-level parallelism the pipe latency needs), the launch size (a launch that is not a
whole number of resident waves pays a partial last round) and the architecture. One fixed geometry
is therefore not a peak probe: 8 chains over 2^20 threads read 91 % of the vector FP64 spec on an
H100 and 54 % on an A100, both at full boost clock. `peak_flops_probe` sweeps a small grid — chains
∈ {4, 8, 16} × launch ∈ {1, 2, 4} × the device's resident-thread capacity (`gpu_sm_count ×
gpu_max_threads_per_sm`) — with one calibrated launch per point, then re-measures the winner as
the best of `trials`. The `PeakProbe` records the winning `chains`, `n_threads`, `n_iters`, the
`capacity`, and the whole sweep, so a manifest says how its denominator was measured. Pin either
axis with an integer or a tuple; a backend without device properties (the CPU backend) gets a
fixed 2^20-thread launch.

A power-managed part reports its power-limited peak: an MI300X holds a lower clock under a
sustained FP64 load than at idle, and the probe measures at that clock. Sample the clock and power
alongside with [`with_gpu_sampler`](telemetry.md) and record them next to the peak; that is the
difference between "the probe reads low" and "the card is power-bound".

`measure_gemm_flops` times `mul!` on the array type's BLAS (cuBLAS, rocBLAS, the host BLAS): on
datacenter parts that is routed to the matrix units and is about twice the vector FP64 rate. It
is the ceiling for what the silicon can do in `T`, useful as a sanity bound on the probe, never as
the denominator for a scalar kernel.

## Cheap probes

```julia
measure_launch_overhead(backend)   # LaunchOverhead: device / enqueue / round-trip µs per launch, queue depth
host_snapshot(backend)             # Julia + BLAS threads vs host cores AND the cgroup quota, memory limits,
                                   # OS kernel, driver / runtime / vendor package versions, warnings as data
first_launch_s(timer)              # the JIT-carrying first launch per device; launch_times(timer; skip_first = true)
```

The launch probe sets the floor below which launches dominate (it differs by an order of magnitude
between drivers, and between a VM and bare metal). The host snapshot is the first section of any
report: two manifests written months apart compare only if they say what they ran on, and the
"pod sees the node's cores while the cgroup grants a fraction" trap is invisible in every device
counter.
