using GPUDiagnostics
using Documenter

DocMeta.setdocmeta!(GPUDiagnostics, :DocTestSetup, :(using GPUDiagnostics); recursive = true)

makedocs(;
    modules = [GPUDiagnostics],
    authors = "Sebastian Micluța-Câmpeanu and contributors",
    sitename = "GPUDiagnostics.jl",
    format = Documenter.HTML(;
        canonical = "https://SebastianM-C.github.io/GPUDiagnostics.jl",
        edit_link = "main",
        assets = String[],
        # the API page autodocs the whole module; the rocprofv3 docstrings alone pass 200 KiB
        size_threshold = 600 * 2^10,
        size_threshold_warn = 400 * 2^10,
    ),
    pages = [
        "Home" => "index.md",
        "Capabilities" => "capabilities.md",
        "Device API and kernel timing" => "device_timing.md",
        "Telemetry" => "telemetry.md",
        "Measured peak and cheap probes" => "peaks_probes.md",
        "Resource report and instruction mix" => "resources_mix.md",
        "Hardware counters" => "hw_counters.md",
        "Report layer" => "report.md",
        "Porting a backend" => "porting.md",
        "Caveats" => "caveats.md",
        "API reference" => "api.md",
    ],
    warnonly = [:missing_docs],
)

deploydocs(; repo = "github.com/SebastianM-C/GPUDiagnostics.jl", devbranch = "main", push_preview = true)
