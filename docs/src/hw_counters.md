# Hardware counters

```julia
cmd = rocprof_command(`julia --project run.jl`; counters = :sq_issue, dir = "prof", name = "cell")   # ROCPROF_COUNTER_SETS
run(cmd)
rc = rocprof_counters("prof"; name = "cell", slots = n_work_items * n_iterations_per_item)  # one user kernel ⇒ auto-selected
rocprof_derived(rc)            # per-slot instruction counts, unit-busy fractions, achieved occupancy, clock
rocprof_summary(rc)            # flat Dict: medians + spreads across dispatches + the derived metrics
diagnostics_dict(rc; prefix = "rocprof_")   # the same, prefixed and schema-tagged for a manifest
```

There is no in-process counter API on AMD, so the wrapper runs the workload under `rocprofv3 --pmc`
(one counter set per pass; `ROCPROF_COUNTER_SETS` lists the sets that fit a single pass on gfx942)
and parses the CSV it writes — pure Julia, no GPU needed for the parsing (the tests run on trimmed
real MI300X collections). The derived metrics use the same normalisation as rocprofv3's own derived
counters: `GRBM_GUI_ACTIVE` is summed over the dies, so every per-cycle rate divides by `n_xcd`
(8 on the MI300X), and the SQ counters are in quad-cycles. Pass `kernel = "…"` (substring or
`Regex`) when the run dispatched more than one user kernel.

NVIDIA has no counterpart in the package yet; the Nsight Compute wrapper and a unified
per-dispatch counter API are tracked in the repository's issues. In-sample counters (NVIDIA GPM)
belong to the [telemetry](telemetry.md) layer instead.
