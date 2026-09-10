# Resource report and instruction mix

```julia
cks = compiled_kernels(backend; pattern = r"_my_driver!")   # kernels this process compiled
r = kernel_resources(backend, only(cks))                    # at the kernel's static workgroup size
r.registers, r.local_mem_bytes, r.shared_mem_bytes           # per-thread regs, spill/stack bytes, LDS/block (missing if unreported)
r.occupancy.active_blocks_per_sm, r.occupancy.fraction       # the runtime's occupancy calculator (missing without one)
r.isa                                                        # AMD: sgpr/vgpr/spill counts, compiler occupancy
r                                                            # prints as a small table
```

Both vendor packages cache every kernel instance the process compiles, so the inventory reaches
kernels that are closures inside driver functions (an AcceleratedKernels `foreachindex` body, say)
without wrapping, recompiling or modifying them; on Julia ≥ 1.12 the closure type carries the
enclosing function's name, which is what `pattern` matches. `shared_mem_bytes` is what the kernel
descriptor *reserves* — LLVM's AMDGPU backend promotes private arrays it cannot keep in registers
to LDS, sized for the kernel's maximum block size, and that reservation, not the source, is what
caps the resident blocks per CU. On NVIDIA the report also re-runs CUDA.jl's bundled `ptxas --verbose` on the
regenerated module PTX, which separates the call-ABI stack frame from true register spills and lists the device
functions assembled out of line (what `CUDABackend(always_inline = true)` removes). Nothing is launched; the ISA
dumps cost a few seconds of compiler time.

## Static instruction mix (native or cross-compiled)

```julia
m = kernel_instruction_mix(backend, only(cks))                 # this device's code
m.counts.fp64_fma, m.counts.fp64_add, m.counts.fp64_mul         # whole binary, every path once
m.hot_loop.counts, m.hot_loop_confidence                        # the per-slot loop: one pass, nested loops once
m942 = kernel_instruction_mix(backend, only(cks); target = "gfx942")   # the MI300X code, without an MI300X
m90 = kernel_instruction_mix(backend, only(cks); target = "sm_90")     # the H100 code, from any CUDA context
fp64_issue_floor(m; n_slots = n_work_items * n_iterations_per_item, peak_fp64_flops = measure_peak_flops(backend),
    kernel_time_s = median_launch_s)                            # FP64-pipe time floor and fraction
```

The disassembly (AMD ISA text from `code_native`; NVIDIA SASS from CUDA.jl's bundled `nvdisasm` on
the cubin) is counted by class — FP64 fma / add / mul / transcendental seeds / other / CDNA packed,
FP32, integer, scalar, memory loads / stores / atomics, constant loads, LDS, control, waits, nops,
other (`MIX_CLASSES`) — for the whole kernel and for every loop of its control-flow graph (natural loops
from dominators), the hot loop being the outermost loop with the most instructions. On AMD the
loop nest is checked against the LLVM assembly printer's own loop annotations (`:high` confidence
when they agree); SASS has none, so the CFG result stands alone (`:medium` with a single dominant
outer loop). `target = "gfx942"` / `"sm_90"` compiles the same kernel — same function, argument
types, `always_inline`, static workgroup size — for another architecture through GPUCompiler, so
the code a rented machine will run can be read before renting it (e.g. whether the CDNA3 compiler
emits `v_pk_fma_f64` at all — for the scalar FP64 kernels this was developed on it does not; or
that ptxas pads an sm_120 hot loop with NOPs where the sm_90 loop has none). Static counts count
code, not execution; against Nsight Compute on an RTX 5090 the hot-loop DFMA / DADD+DMUL counts of
an FP64 Newton-iteration kernel reproduced the measured per-slot warp-instruction counts exactly.
`fp64_issue_floor` divides a per-slot FP64 count × executed slots by the
measured FP64 lane-instruction rate (`measure_peak_flops` / 2 — an FMA is 2 FLOP): the time the
FP64 pipe alone needs per launch, assuming every FP64 instruction issues at the FMA rate.

The classifiers are ordered rule tables, `SASS_RULES` / `AMD_RULES :: Vector{Pair{Regex, Symbol}}`,
built on the mnemonic grammars (SASS: the leading operand-type letter `D`/`F`/`H`/`I`/`U`, the
`LD`/`ST`/`ATOM`/`RED` + `G`/`L`/`S`/`C` memory suffixes; AMD: the `v_`/`s_`/`ds_`/`global_` prefixes and
`_f64`/`_f32` suffixes) with a short exception list ahead of each generic rule; the last rule is a
catch-all whose hits are `other` but reported as `unclassified` / `unclassified_opcodes` / `coverage`
(1.0 on all four validated targets). The vectors are mutable — `pushfirst!(SASS_RULES, r"^MYOP" =>
:int)` overrides for the session. `kernel_ir_mix` (also the `ir` field of `kernel_instruction_mix`)
walks the OPTIMIZED LLVM IR of the same job with LLVM.jl, typed by opcode and operand type (with the `contract` fast-math flag counted as `fp64_contract`: Julia lowers `muladd` to a
contract-flagged `fmul`/`fadd` pair, which the AMD backend fuses only at instruction selection)
(`IR_CLASSES`: fp64 fma / add / mul / div / neg / sqrt / cmp / cvt / intrinsic, fp32, int, memory,
call, control, other) — the arithmetic before the backend: IR `fma` 0 against 132 `v_fma_f64` /
301 DFMA is the backend's FMA contraction, IR `div` 14 / `sqrt` 3 against the machine's
reciprocal seeds + FMA sequences its expansion. Whole-module totals (no IR loop attribution).

Gotchas: CUDA.jl's `code_sass` loads the module on the current device (via CUPTI), which an
`sm_90` cubin cannot do on an `sm_120` GPU — the extension compiles with `CUDACore.compile` and runs
`nvdisasm` on the image directly, so no CUPTI and no device of the target architecture are needed.
AMDGPU.jl caches the OCLC ISA-version device library under an ISA-agnostic key; the extension evicts
it around a cross-compile so gfx942 links `oclc_isa_version_942`, not the current device's.
