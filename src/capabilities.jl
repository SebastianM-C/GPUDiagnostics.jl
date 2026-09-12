# ── Backend capabilities ─────────────────────────────────────────────────────────────────────
#
# Not every backend can answer every question: the KA CPU backend has devices and a host
# clock but no telemetry; a vendor without a public ISA has an IR mix but no native one; a
# device without FP64 has no FP64 peak. The capability trait says which subsystems a backend
# implements, so callers can branch on it, reports can say "not reported by this backend"
# instead of failing, and the conformance suite can check that a declared feature works and an
# undeclared one fails with `BackendUnsupported` rather than a `MethodError` or a vendor-named
# string. Extensions declare `supports(::TheirBackend, ::Val{:feature}) = true` per feature.

"""
    FEATURES

The capability names a backend may declare through [`supports`](@ref), each naming one
subsystem of the package:

| feature | entry points |
|---|---|
| `:devices` | `gpu_device_count`, `gpu_device`, `gpu_device!`, `gpu_name`, `gpu_arch` |
| `:device_props` | `gpu_sm_count`, `gpu_max_threads_per_sm`, `gpu_memory_info`, `thread_fill_occupancy` |
| `:events` | `gpu_event`, `gpu_elapsed`, `LaunchTimer` |
| `:telemetry` | `gpu_power`, `gpu_utilization`, `gpu_sample`, `gpu_sampler_sources`, `with_gpu_sampler` |
| `:telemetry_counters` | hardware counters in the sample (NVIDIA GPM: achieved occupancy, pipe utilizations, …) |
| `:peak_flops` | `measure_peak_flops` (the FMA-chain probe runs on this backend) |
| `:fp64` | the device computes in `Float64`: `measure_peak_flops(b, Float64)`, `fp64_issue_floor` |
| `:kernel_inventory` | `compiled_kernels` |
| `:resources` | `kernel_resources` (registers, spill/local and shared memory) |
| `:occupancy` | the runtime occupancy calculator behind `kernel_resources` |
| `:native_mix` | `kernel_instruction_mix` (a disassembly of the device code exists) |
| `:ir_mix` | `kernel_ir_mix` |
| `:hw_counters` | `hw_counter_command`, `hw_counter_status` (rocprofv3 on AMD, ncu on NVIDIA) |
"""
const FEATURES = (:devices, :device_props, :events, :telemetry, :telemetry_counters, :peak_flops, :fp64,
    :kernel_inventory, :resources, :occupancy, :native_mix, :ir_mix, :hw_counters)

"""
    supports(backend, feature::Symbol) -> Bool
    supports(backend, ::Val{feature}) -> Bool

Whether `backend` implements the subsystem `feature` (one of [`FEATURES`](@ref)). The default
is `false`; the KA `CPU` backend declares `:devices`, `:events`, `:peak_flops`, `:fp64` and
`:kernel_inventory` (an empty inventory), and each vendor extension declares what its runtime
can answer. A backend extension adds a method per feature:

    GPUDiagnostics.supports(::MyBackend, ::Val{:telemetry}) = true

An entry point of an undeclared feature throws [`BackendUnsupported`](@ref).
"""
supports(::Backend, ::Val) = false
supports(backend::Backend, feature::Symbol) = supports(backend, Val(feature))

supports(::KA.CPU, ::Val{:devices}) = true
supports(::KA.CPU, ::Val{:events}) = true
supports(::KA.CPU, ::Val{:peak_flops}) = true
supports(::KA.CPU, ::Val{:fp64}) = true
supports(::KA.CPU, ::Val{:kernel_inventory}) = true

"""    capabilities(backend) -> Vector{Symbol}

The features of [`FEATURES`](@ref) that `backend` declares through [`supports`](@ref), in
`FEATURES` order."""
capabilities(backend::Backend) = Symbol[f for f in FEATURES if supports(backend, Val(f))]

# Backend type name ⇒ the package whose extension provides its methods. Consulted for the
# error message only, so it may list backends whose extension does not exist yet.
const BACKEND_PROVIDERS = Dict(
    "CUDABackend" => "CUDA.jl",
    "ROCBackend" => "AMDGPU.jl",
    "MetalBackend" => "Metal.jl",
    "oneAPIBackend" => "oneAPI.jl",
)

"""
    BackendUnsupported(backend, feature, entry)

Thrown by an entry point (`entry`, the function's name) when `backend` does not declare
`feature` through [`supports`](@ref): either the backend cannot provide it (the CPU backend
has no telemetry, a device without FP64 has no FP64 peak) or the vendor extension that would
provide it is not loaded — the message says which package that is when it is known.
"""
struct BackendUnsupported <: Exception
    backend::Any
    feature::Symbol
    entry::Symbol
end

function Base.showerror(io::IO, e::BackendUnsupported)
    T = string(nameof(typeof(e.backend)))
    print(io, e.entry, ": ", T, " does not support :", e.feature)
    e.feature in FEATURES || print(io, " (not a GPUDiagnostics feature; see FEATURES)")
    pkg = get(BACKEND_PROVIDERS, T, nothing)
    if pkg !== nothing && isempty(capabilities(e.backend))
        print(io, " — load ", pkg, " to enable its extension")
    elseif pkg !== nothing
        print(io, " (", pkg, " is loaded; this vendor cannot answer it)")
    end
    return nothing
end

# Convention for values a backend cannot answer, at the API boundary (every NamedTuple / struct
# field and dict value the package returns): `missing`. `NaN` only inside the numeric telemetry
# matrix and on its TSV wire, where rows must stay numeric. `nothing` means the thing does not
# exist or was not asked for (a kernel with no loop, `ir = false`, no trace file) — never "the
# vendor could not report it". Serialisation (diagnostics_dict) omits `missing` keys.

# Guard for entry points: no-op when declared, else the typed error.
@inline function _require(backend::Backend, feature::Symbol, entry::Symbol)
    supports(backend, Val(feature)) || throw(BackendUnsupported(backend, feature, entry))
    return nothing
end
