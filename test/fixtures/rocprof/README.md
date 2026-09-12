# rocprofv3 fixtures

Real collections, trimmed to a few dispatches; nothing here needs a GPU to parse.

## `sq1_*`, `sq2_*`, `c2a_*` — gfx942 (MI300X VF), rocprofv3 1.1.0 / ROCm 7.2.4

The production field kernel of the first consumer (an AcceleratedKernels `foreachindex`, so
`gpu__forindices_global_(…)`) plus its FMA-chain peak probe: `sq2` is the instruction-issue set
with agent info and kernel trace, `sq1` the wave-residency set without agent info (device values
must come from `device_overrides`), `c2a` the L1-pipe set. The normalisation tests
(`GRBM_GUI_ACTIVE / 8` dies, quad-cycle SQ counters, `*_sum / (cycles × 304)`) are built on them.

## `w7900_*` — gfx1100 (Radeon Pro W7900), rocprofv3 1.1 / ROCm 7.2.4, `STABLE_STD`

One KernelAbstractions FP64 FMA-chain kernel, 262144 work-items × 100000 dependent FMAs
(8192 wave32 waves, ~148 ms per dispatch at the pinned 959 MHz shader clock); dispatch 2 is a
JIT launch with 8 FMAs per work-item. Collected inside a ROCm container with device-qualified
counters (`GRBM_GUI_ACTIVE:device=0 …`) because the host's integrated GPU is enumerated too.
The agent info is trimmed to the W7900's row, with the PCI location, DRM minor, GPU / hive ids and
firmware versions zeroed; the architectural columns the parser reads are untouched.
Validates the RDNA entries of the SQ layout table: `sq_cycle_unit = 1` keeps resident waves
(1346) below the 96 × 32 hardware maximum, `sq_instances = 48` WGPs puts `SQ_BUSY_CYCLES` at
0.89 of the dispatch, and `SQ_INSTS_VALU × 32 / slots` is 1.0001 per FMA. `SQ_WAVES` reads
116097 instead of 8192 on the last dispatch — a counter glitch observed in three of five W7900
collections, always on one dispatch.
