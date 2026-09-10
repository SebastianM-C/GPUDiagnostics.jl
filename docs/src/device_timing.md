# Device API and kernel timing

## Device API

```julia
using GPUDiagnostics, CUDA          # or AMDGPU
backend = CUDABackend()

gpu_device_count(backend), gpu_name(backend), gpu_arch(backend)
gpu_sm_count(backend) * gpu_max_threads_per_sm(backend)   # resident-thread capacity
gpu_memory_info(backend), gpu_power(backend), gpu_utilization(backend)
thread_fill_occupancy(backend, n_threads)                 # static upper bound, see Caveats
```

Everything dispatches on the KernelAbstractions `Backend`. KernelAbstractions has no device
management of its own; `gpu_device!` is how a multi-device driver pins each shard's task to a GPU
(1-based across vendors).

When launches are queued asynchronously, a host clock around a launch measures enqueue latency.
An event recorded on the launch stream fires when the GPU reaches it in stream order, so a pair
around a launch brackets exactly that kernel — two barrier packets per launch, no host stall.

```julia
timer = LaunchTimer()
lane = launch_lane(timer, backend)          # once per loop (device id)
for item in work
    e0 = launch_tick(timer, backend)
    my_kernel!(backend)(item; ndrange = n)  # asynchronous
    launch_tock!(timer, lane, backend, e0)
end
launch_times(timer)                          # Dict(device => [seconds per launch, …])
```

Pass `nothing` instead of a timer and the hooks are no-ops. The timer is safe to share across
per-device tasks (pushes are locked; events are per-stream).
