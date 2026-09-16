# amdgpu `gpu_metrics` fixtures

`w7900_gpu_metrics_idle.bin` — the 120-byte `gpu_metrics` v1.3 snapshot of an idle Radeon Pro
W7900 (RDNA3, gfx1100), read from `/sys/class/drm/card<N>/device/gpu_metrics` on 2026-09-13:
9 W, 3 MHz gfx clock, 0 % activity, `throttle_status = 2`, `indep_throttle_status = 1 << 36`
(`:temp_hotspot`, the bit RDNA3 reports permanently). Device telemetry only; the layout is the
kernel's `struct gpu_metrics_v1_3`.

`mi300x_gpu_metrics_v1_9_idle.bin` — the 1150-byte `gpu_metrics` v1.9 snapshot of an idle
MI300X (CDNA3, gfx942) read on 2026-09-16 from a cloud SR-IOV virtual function (ROCm 7.2.4,
amdgpu DKMS 6.16.13, guest kernel 6.8): a self-describing attribute table of 47 entries —
hotspot 42 °C, 194 W socket power, 0 % activity, eight per-XCD gfx clocks at 2098–2115 MHz,
`accumulation_counter = 54536484`, `ppt_residency_acc = 633093` (1.2 % lifetime PPT residency),
every thermal residency 0, plus the per-XCD busy and below-host-limit accumulators. Device
telemetry only; v1.9 carries no serials, node ids or PCI addresses beyond link width/speed.
