# NVIDIA counter fixtures

`fp64_ncu.csv` is a trimmed raw-page export from Nsight Compute 2025.4.1 on a
GPU with compute capability 12.0. Two launches of the `counter_probe!` kernel each run 4,096 threads
with 32 FP64 FMAs per thread. Each reports 131,072 predicated-on FMA thread
instructions, zero DADD/DMUL instructions, and three profiler replay passes.
Clock control was `none`; cache control and replay mode were ncu's defaults
(`all` and `kernel`). The durations are profiler measurements, not benchmarks.

Only selected metric columns and launch identity/geometry remain. Process,
context, and stream identifiers were replaced with fixed test values. Host,
process-name, path, and topology fields and tool log messages were removed.
The marketing name was replaced by a generic label. Numeric counters, units,
durations, and kernel name are unchanged.
