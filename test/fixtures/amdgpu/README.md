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

`mi300x_gpu_metrics_v1_6_dkms_idle.bin` — the 1664-byte `gpu_metrics` v1.6 snapshot of an idle
MI300X read on 2026-09-16 in a container on a bare-metal host running AMD's DKMS amdgpu 6.10.5
(ROCm 6.3-era driver, ROCm 7.2.4 userspace). The DKMS v1.6 is the upstream struct with
`num_partition` and `xcp_stats[NUM_XCP]` appended, hence 1664 bytes instead of 312, at the same
leading offsets: hotspot 42 °C, memory 40 °C, 120 W, 1 % gfx activity, eight XCD clocks at
158–160 MHz, `accumulation_counter = 10657785`. Firmware quirk recorded on purpose: the PPT,
socket-thermal and HBM-thermal residencies all EQUAL the accumulation counter at idle (a
nominal 100 % residency), so this driver/firmware pair's residency fractions need a delta check
under known load before they are believed. Device telemetry only.

`mi300x_gpu_metrics_v1_9_load_{a,b}.bin` — two v1.9 snapshots from the same virtual function as
the idle capture, 5 s apart, while an FP64 field kernel ran at 100 % gfx activity at 705–707 W
against the 750 W cap: `accumulation_counter` 55257356 → 55262364, `ppt_residency_acc` 749905 →
754247 (Δ 4342 / 5008 = 86.7 % of the interval under the package-power limiter), every thermal
residency 0, hotspot 59 → 68 °C, the eight XCD clocks in 1262–1317 MHz. The pair is what
`power_violation_fraction` is computed from. Device telemetry only.
