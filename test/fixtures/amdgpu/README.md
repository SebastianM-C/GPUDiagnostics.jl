# amdgpu `gpu_metrics` fixtures

`w7900_gpu_metrics_idle.bin` — the 120-byte `gpu_metrics` v1.3 snapshot of an idle Radeon Pro
W7900 (RDNA3, gfx1100), read from `/sys/class/drm/card<N>/device/gpu_metrics` on 2026-09-13:
9 W, 3 MHz gfx clock, 0 % activity, `throttle_status = 2`, `indep_throttle_status = 1 << 36`
(`:temp_hotspot`, the bit RDNA3 reports permanently). Device telemetry only; the layout is the
kernel's `struct gpu_metrics_v1_3`.
