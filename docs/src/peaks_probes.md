# Measured peak and cheap probes

```julia
measure_peak_flops(backend)            # FP64 FLOP/s, dependent-FMA-chain kernel, best of 5
measure_peak_flops(backend, Float32)   # the FP32 rate of the same probe
```

No per-architecture table: the probe measures the attainable vector rate at the clocks the device
actually holds, and is never routed to matrix/tensor units (the wrong yardstick for scalar kernels).
The result is checked against a host reference so a mis-launched kernel cannot be credited. It runs on
the CPU backend too, one scalar chain per work-item, which under-reports the host by its SIMD width;
the vectorised host number is `LinearAlgebra.peakflops(2048; ntrials = 3)`, a different quantity.

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
