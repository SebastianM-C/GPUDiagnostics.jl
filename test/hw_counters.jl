using Test, GPUDiagnostics
import Statistics

struct CounterTestBackend{C <: HWCounterCollector} <: GPUDiagnostics.KA.Backend end
GPUDiagnostics.supports(::CounterTestBackend, ::Val{:hw_counters}) = true
GPUDiagnostics.backend_counter_collector(::CounterTestBackend{C}) where {C} = C()

# Records what `hw_counter_window` does to the profiling range; the device is task-local, as in CUDA.jl.
struct WindowBackend <: GPUDiagnostics.KA.Backend
    log::Vector{Any}
    fail::Base.RefValue{Symbol}           # the hook that throws
    gates::Dict{Int, Base.Event}          # device => event its range waits for before it opens
end
WindowBackend() = WindowBackend([], Ref(:none), Dict())
struct OtherWindowBackend <: GPUDiagnostics.KA.Backend
    inner::WindowBackend
end
const _Windowed = Union{WindowBackend, OtherWindowBackend}
_window_log(b::WindowBackend) = b
_window_log(b::OtherWindowBackend) = b.inner
GPUDiagnostics.supports(::_Windowed, ::Union{Val{:counter_window}, Val{:devices}}) = true
GPUDiagnostics.gpu_device(::_Windowed) = get(task_local_storage(), :window_device, 1)
GPUDiagnostics.gpu_device!(b::_Windowed, i::Integer) = (p = gpu_device(b); task_local_storage(:window_device, Int(i)); p)
GPUDiagnostics.KA.synchronize(b::_Windowed) = (push!(_window_log(b).log, :sync); nothing)
function GPUDiagnostics.backend_counter_window_start(b::_Windowed)
    w = _window_log(b)
    w.fail[] === :start && error("no range")
    haskey(w.gates, gpu_device(b)) && wait(w.gates[gpu_device(b)])
    sleep(0.02)   # another task may try to open a window meanwhile
    push!(w.log, (:start, gpu_device(b)))
    return nothing
end
function GPUDiagnostics.backend_counter_window_stop(b::_Windowed)
    w = _window_log(b)
    push!(w.log, (:stop, gpu_device(b)))
    w.fail[] === :stop && error("range not closed")
    return nothing
end
# A regression must fail, not hang the suite.
_finished(ts...) = timedwait(() -> all(istaskdone, ts), 20) === :ok && all(!istaskfailed, ts)

@testset "hardware counters" begin
    fx = joinpath(@__DIR__, "fixtures", "rocprof")
    slots = 401 * 401 * 1666
    @test_throws ArgumentError hw_counters(fx)
    @test_throws ArgumentError hw_counters(fx; name = "sq2")
    @test_throws ArgumentError hw_counters(fx; name = "missing")
    @test_throws ArgumentError hw_counters(fx; name = "sq2", kernel = "absent")
    @test_throws ArgumentError hw_counters(fx; name = "sq2", kernel = "forindices", slots = 0)
    @test_throws ArgumentError hw_counters(fx; name = "sq2", kernel = "forindices", slots = [1, 2])
    @test_throws ArgumentError hw_counters(fx; name = "sq2", kernel = "forindices", required_metrics = ["absent"])
    h = hw_counters(fx; name = "sq2", kernel = "forindices", slots,
        provenance = Dict("tool_version" => "1.1.0", "hostname" => "excluded", "command" => "excluded"))
    @test h isa HWCounters && h.vendor === :amd && h.tool === :rocprofv3 && length(h) == 3
    @test [d.id for d in h.dispatches] == [2, 3, 4]
    @test h["GRBM_GUI_ACTIVE"] == [575399494.0, 570161410.0, 537860558.0]
    @test h["SQ_INSTS_VMEM_RD"] == [365118475.0, 365365297.0, 364581971.0]
    @test [d.duration_s for d in h.dispatches] ≈ [0.040019311, 0.039977577, 0.040177312]
    @test all(d -> d.device["n_cu"] == 304 && d.device["n_xcd"] == 8, h.dispatches)
    @test all(d -> d.resources["vgpr_count"] == 64 && d.resources["scratch_bytes"] == 92, h.dispatches)
    @test all(ismissing, values(h.units))
    @test h.provenance == Dict("tool_version" => "1.1.0")
    @test Set(keys(h)) == Set(COUNTER_SETS[:amd][:issue].metrics)
    @test_throws KeyError h["unknown"]
    derived = hw_counter_derived(h)
    @test derived["amd_insts_per_slot_vmem_rd"] ≈ h["SQ_INSTS_VMEM_RD"] .* 64 ./ slots
    @test derived["amd_active_clock_GHz"] ≈ h["GRBM_GUI_ACTIVE"] ./ 8 ./ [d.duration_s for d in h.dispatches] ./ 1e9
    @test !haskey(derived, "clock_GHz") # different averaging windows are not silently unified
    s = hw_counter_summary(h)
    @test s["raw_GRBM_GUI_ACTIVE_median"] == 570161410
    @test s["derived_amd_insts_per_slot_vmem_rd_median"] ≈ Statistics.median(derived["amd_insts_per_slot_vmem_rd"])
    @test s["dispatch_s_samples"] == 3 && s["resources_vgpr_count"] == 64
    @test s["collection_tool_version"] == "1.1.0" && !haskey(s, "collection_hostname")
    dd = diagnostics_dict(h; prefix = "hw_")
    @test dd["gpudiagnostics_schema"] == GPUDIAGNOSTICS_SCHEMA
    @test dd["hw_raw_GRBM_GUI_ACTIVE_median"] == 570161410
    @test all(k -> k == "gpudiagnostics_schema" || startswith(k, "hw_"), keys(dd))
    @test occursin("rocprofv3", sprint(show, h))
    varied = hw_counters(fx; name = "sq2", kernel = "forindices", slots = [slots, 2slots, slots])
    @test hw_counter_derived(varied)["amd_insts_per_slot_vmem_rd"][2] ≈ derived["amd_insts_per_slot_vmem_rd"][2] / 2
    @test !haskey(hw_counter_summary(varied), "slots")

    # rocprofv3 1.4 (ROCm 10.1) on a two-socket MI350X host: `Agent 2` is the GPU's
    # Logical_Node_Id; its Node_Id is 4 (the CPUs take 0 and 1)
    m350 = hw_counters(fx; name = "mi350x")
    d350 = first(m350.dispatches).device
    @test d350["architecture"] == "gfx950" && d350["wave_size"] == 64 && d350["n_cu"] == 256 && d350["n_xcd"] == 8
    @test m350["SQ_WAVES"] == [64.0, 64.0, 64.0]

    l1 = hw_counters(fx; name = "c2a")
    e = hw_counter_derived(l1)
    @test e["amd_td_busy"] ≈ l1["TD_TD_BUSY_sum"] ./ (l1["GRBM_GUI_ACTIVE"] ./ 8 .* 304)
    @test e["amd_l1_miss"] ≈ l1["TCP_TCC_READ_REQ_sum"] ./ l1["TCP_TOTAL_CACHE_ACCESSES_sum"]
    w0 = hw_counters(fx; name = "sq1")
    @test all(ismissing, hw_counter_derived(w0)["amd_resident_waves"])
    devid = first(w0.dispatches).device_id
    override = Dict(devid => Dict("architecture" => "gfx942", "n_cu" => 304, "n_xcd" => 8,
        "wave_size" => 64, "max_waves_per_cu" => 32, "n_se" => 32))
    w = hw_counters(fx; name = "sq1", device_overrides = override)
    e = hw_counter_derived(w)
    @test e["amd_resident_waves"] ≈ 4 .* w["SQ_WAVE_CYCLES"] ./ (w["GRBM_GUI_ACTIVE"] ./ 8)
    @test e["amd_elapsed_occupancy"] ≈ e["amd_resident_waves"] ./ 304 ./ 32
    @test e["amd_wave_wait_frac"] ≈ w["SQ_WAIT_ANY"] ./ w["SQ_WAVE_CYCLES"]
    @test !haskey(e, "achieved_occupancy")
    @test all(d -> d.device["sq_cycle_unit"] == 4, h.dispatches)   # gfx942 agent info → CDNA quad-cycles
    @test s["device_sq_cycle_unit"] == 4 && s["device_sq_instances"] == 32 && s["device_architecture"] == "gfx942"
    override[devid]["architecture"] = "gfx1100"   # RDNA: AMD's OccupancyPercent uses 100·, not 400·
    e1 = hw_counter_derived(hw_counters(fx; name = "sq1", device_overrides = override))
    @test e1["amd_resident_waves"] ≈ e["amd_resident_waves"] ./ 4 && e1["amd_wave_cycles"] ≈ e["amd_wave_cycles"] ./ 4
    @test e1["amd_active_clock_GHz"] ≈ e["amd_active_clock_GHz"]   # unit-independent
    @test e1["amd_sq_busy"] ≈ e["amd_sq_busy"] .* 32 ./ 152        # SQ_BUSY_CYCLES: per SE (32) on CDNA, per WGP (304 ÷ 2) on RDNA
    override[devid]["architecture"] = "gfx908"    # not in the table → the SQ unit is unknown, the rest stands
    e2 = hw_counter_derived(hw_counters(fx; name = "sq1", device_overrides = override))
    @test all(ismissing, e2["amd_resident_waves"]) && all(ismissing, e2["amd_wave_cycles"]) && all(ismissing, e2["amd_sq_busy"])
    @test e2["amd_active_clock_GHz"] ≈ e["amd_active_clock_GHz"] && e2["amd_wave_wait_frac"] ≈ e["amd_wave_wait_frac"]
    override[devid]["sq_cycle_unit"] = 4          # ... unless the caller supplies them
    override[devid]["sq_instances"] = 32
    e3 = hw_counter_derived(hw_counters(fx; name = "sq1", device_overrides = override))
    @test e3["amd_resident_waves"] ≈ e["amd_resident_waves"] && e3["amd_sq_busy"] ≈ e["amd_sq_busy"]
    @test hw_counter_summary(hw_counters(fx; name = "sq1", device_overrides = override))["device_sq_instances"] == 32
    override[devid]["sq_cycle_unit"] = 0
    @test_throws ArgumentError hw_counters(fx; name = "sq1", device_overrides = override)

    @testset "gfx1100 capture: RDNA SQ layout" begin
        S = 262144 * 100000
        w = hw_counters(fx; name = "w7900", kernel = "fma_probe", slots = [262144 * 8, S, S])
        dev = first(w.dispatches).device
        @test dev["architecture"] == "gfx1100" && dev["wave_size"] == 32 && dev["n_cu"] == 96 && dev["n_xcd"] == 1
        @test dev["sq_cycle_unit"] == 1 && dev["sq_instances"] == 48   # per WGP, not per shader engine (6)
        d = hw_counter_derived(w)
        @test d["amd_insts_per_slot_valu"][2] ≈ 1.00012 rtol = 1e-4        # one v_fma_f64 per slot + 12 VALU per work-item
        @test d["amd_insts_per_slot_valu"][1] ≈ (8 + 12) / 8 rtol = 1e-3
        @test all(x -> 0.89 < x < 0.90, d["amd_sq_busy"][2:3])            # 6 shader engines would give 7.1
        @test all(x -> 1300 < x < 1400, d["amd_resident_waves"][2:3])     # ×4 would exceed the 3072 hardware maximum
        @test all(x -> x < 1, d["amd_elapsed_occupancy"]) && d["amd_waves_per_cu"][2] ≈ d["amd_resident_waves"][2] / 96
        @test all(x -> 1.02 < x < 1.04, d["amd_active_clock_GHz"][2:3])   # 959 MHz shader clock pinned by STABLE_STD; GRBM counts ~7 % faster
        @test w["SQ_WAVES"][1:2] == [8192, 8192] && w["SQ_WAVES"][3] == 116097   # the glitch stays visible in the raw data
        @test hw_counter_summary(w)["raw_SQ_WAVES_median"] == 8192
    end

    @testset "directory detection stops at the first header line" begin
        mktempdir() do dir
            cp(joinpath(fx, "w7900_counter_collection.csv"), joinpath(dir, "w7900_counter_collection.csv"))
            trace = joinpath(dir, "w7900_kernel_trace.csv")   # a large non-counter CSV next to the collection
            open(trace, "w") do io
                println(io, "\"Kind\",\"Agent_Id\",\"Dispatch_Id\",\"Kernel_Name\",\"Start_Timestamp\"")
                for i in 1:2000; println(io, "KERNEL_DISPATCH,Agent 1,", i, ",k,", i); end
            end
            @test length(hw_counters(dir)) == 3
            @test GPUDiagnostics._counter_csv_vendor(trace) === nothing
            # only the first non-log line is inspected: a header on the second line does not count
            write(joinpath(dir, "late.csv"), "not,a,header\n\"Dispatch_Id\",\"Counter_Name\"\n")
            @test GPUDiagnostics._counter_csv_vendor(joinpath(dir, "late.csv")) === nothing
            write(joinpath(dir, "x_ncu.csv"), "==PROF== Connected\n\n\"ID\",\"Kernel Name\",\"Metric Name\",\"Metric Unit\",\"Metric Value\"\r\n")
            @test GPUDiagnostics._counter_csv_vendor(joinpath(dir, "x_ncu.csv")) === :nvidia
        end
    end

    @testset "dispatch identity and partial metadata" begin
        mktempdir() do dir
            path = joinpath(dir, "probe_counter_collection.csv")
            write(path, "Dispatch_Id,Process_Id,Agent_Id,Kernel_Name,Counter_Name,Counter_Value,Grid_Size\n" *
                "1,10,Agent 1,probe,SQ_WAVES,3,256\n1,11,Agent 2,probe,SQ_WAVES,4,512\n" *
                "2,11,Agent 2,probe,OTHER,0,512\n")
            h = hw_counters(path)
            @test length(h) == 3 && [d.process_id for d in h.dispatches] == [10, 11, 11]
            @test isequal(h["SQ_WAVES"], [3.0, 4.0, missing])
            @test all(d -> ismissing(d.duration_s), h.dispatches)
            @test !haskey(hw_counter_summary(h), "resources_grid_size")
            @test !haskey(hw_counter_summary(h), "dispatch_total_s")
            @test hw_counter_summary(h)["raw_SQ_WAVES_samples"] == 2
            open(path, "a") do io
                println(io, "1,10,Agent 1,probe,SQ_WAVES,5,256")
            end
            @test_throws ArgumentError hw_counters(path)
        end
    end

    @testset "NVIDIA long and wide CSV, units, missing values" begin
        mktempdir() do dir
            long = joinpath(dir, "probe_ncu.csv")
            write(long, "==PROF== Connected\n" *
                "\"ID\",\"Process ID\",\"Kernel Name\",\"Context\",\"Stream\",\"Device\",\"Metric Name\",\"Metric Unit\",\"Metric Value\"\n" *
                "0,10,\"probe(a, b)\",1,7,0,gpu__time_duration.sum,nsecond,1000\n" *
                "0,10,\"probe(a, b)\",1,7,0,smsp__sass_thread_inst_executed_op_dfma_pred_on.sum,inst,\"1,024\"\n" *
                "0,10,\"probe(a, b)\",1,7,0,smsp__sass_thread_inst_executed_op_dadd_pred_on.sum,inst,128\n" *
                "0,10,\"probe(a, b)\",1,7,0,smsp__sass_thread_inst_executed_op_dmul_pred_on.sum,inst,64\n" *
                "0,10,\"probe(a, b)\",1,7,0,sm__warps_active.avg.pct_of_peak_sustained_active,%,50\n" *
                "1,10,\"probe(a, b)\",1,7,0,gpu__time_duration.sum,nsecond,2000\n" *
                "1,10,\"probe(a, b)\",1,7,0,smsp__sass_thread_inst_executed_op_dfma_pred_on.sum,inst,n/a\n")
            h = hw_counters(long; slots = [128, 256])
            @test length(h) == 2 && h.vendor === :nvidia && h.tool === :ncu
            @test [d.duration_s for d in h.dispatches] ≈ [1e-6, 2e-6]
            @test h.units["gpu__time_duration.sum"] == "nsecond"
            @test first(h.dispatches).queue_id == 7
            d = hw_counter_derived(h)
            @test isequal(d["insts_per_slot_fp64_fma"], [8.0, missing])
            @test d["fp64_flop_per_slot"][1] == 17.5
            @test isequal(d["nvidia_active_occupancy"], [0.5, missing])
            @test hw_counter_summary(h)["derived_insts_per_slot_fp64_fma_samples"] == 1
            @test_throws ArgumentError hw_counters(long; required_metrics = ["smsp__sass_thread_inst_executed_op_dfma_pred_on.sum"])
            # routed metadata keeps the counters' rules: no-value cells are missing, repeats must agree
            hdr = "\"ID\",\"Process ID\",\"Kernel Name\",\"Context\",\"Stream\",\"Device\",\"Metric Name\",\"Metric Unit\",\"Metric Value\"\n"
            meta = joinpath(dir, "meta_ncu.csv")
            write(meta, hdr * "0,1,k,1,1,0,launch__grid_size,,n/a\n0,1,k,1,1,0,launch__block_size,,256\n" *
                "0,1,k,1,1,0,launch__registers_per_thread,register/thread,16\n0,1,k,1,1,0,gpu__time_duration.sum,ns,10\n")
            m = hw_counters(meta)
            @test ismissing(first(m.dispatches).resources["grid_size"]) && ismissing(first(m.dispatches).resources["grid_blocks"])
            @test first(m.dispatches).resources["workgroup_size"] == 256 && first(m.dispatches).resources["registers"] == 16
            open(meta, "a") do io; println(io, "0,1,k,1,1,0,launch__registers_per_thread,register/thread,32"); end
            @test_throws ArgumentError hw_counters(meta)
            # replay passes are provenance of the SELECTED kernel, not of the whole export
            two = joinpath(dir, "two_ncu.csv")
            write(two, hdr * "0,1,warm,1,1,0,profiler__replayer_passes,pass,1\n0,1,warm,1,1,0,gpu__time_duration.sum,ns,5\n" *
                "1,1,hot,1,1,0,profiler__replayer_passes,pass,3\n1,1,hot,1,1,0,gpu__time_duration.sum,ns,7\n")
            @test hw_counters(two; kernel = "hot").provenance["passes"] == 3
            @test hw_counters(two; kernel = "warm").provenance["passes"] == 1
            wide = joinpath(dir, "wide_ncu.csv")
            write(wide, "ID,Process ID,Kernel Name,Device,gpu__time_duration.sum,lts__t_sector_hit_rate.pct\n" *
                ",,,,usecond,%\n0,10,probe,0,2,75\n0,11,probe,1,3,0\n")
            w = hw_counters(:nvidia, wide)
            @test [d.duration_s for d in w.dispatches] ≈ [2e-6, 3e-6]
            @test hw_counter_derived(w)["nvidia_l2_sector_hit_rate"] == [0.75, 0]
            @test_throws ArgumentError hw_counters(dir)
            @test length(hw_counters(dir; name = "wide")) == 2
            open(wide, "a") do io
                println(io, "==ERROR== collection failed")
            end
            @test_throws ArgumentError hw_counters(wide)
            @test_throws ArgumentError GPUDiagnostics._counter_number("bad")
            @test GPUDiagnostics._counter_int("1750000000000000001") == 1750000000000000001
            @test ismissing(GPUDiagnostics._safe_ratio(1, 0))
            @test_throws ArgumentError GPUDiagnostics._csv_fields("\"unterminated")
        end
    end

    @testset "commands and discovery" begin
        for (collector, tool) in ((RocprofV3(), "rocprofv3"), (NsightCompute(), "ncu"))
            backend = CounterTestBackend{typeof(collector)}()
            cmd = setenv(`julia run.jl`, ["TEST_KEY=2"]; dir = "/tmp")
            c = hw_counter_command(backend, cmd; dir = "out", name = "probe", timeout_s = nothing)
            @test c.exec[1:3] == ["/usr/bin/env", "$PROFILER_ENV=$tool", tool]
            @test hw_counter_command(collector, `true`; dir = "out", name = "probe").env === nothing   # ENV is not copied into the Cmd
            @test c.exec == hw_counter_command(collector, cmd; dir = "out", name = "probe", timeout_s = nothing).exec
            @test c.env == cmd.env && c.dir == cmd.dir
            st = hw_counter_status(backend; executable = "gpudiag_no_such_profiler")
            @test st.tool == Symbol(tool) && !st.available && ismissing(st.permitted)
            @test !hw_counters_available(backend; executable = "gpudiag_no_such_profiler")
            @test hw_counter_command(backend, cmd; dir = "out", name = "probe",
                executable = "/custom/profiler", timeout_s = nothing).exec[3] == "/custom/profiler"
        end
        # `env` would read such a path as an assignment; behind `timeout` it is an ordinary argument
        @test_throws ArgumentError hw_counter_command(NsightCompute(), `true`; dir = "out", name = "probe", executable = "/a=b/ncu", timeout_s = nothing)
        @test hw_counter_command(NsightCompute(), `true`; dir = "out", name = "probe", executable = "/a=b/ncu").exec[7] == "/a=b/ncu"
        @test GPUDiagnostics._marked(["ncu", "x"], NsightCompute(); windows = true) == ["ncu", "x"]
        c = hw_counter_command(RocprofV3(), `julia run.jl`; set = :occupancy, dir = "out", name = "probe")
        @test c.exec[3:8] == ["timeout", "-k", "20", "600", "rocprofv3", "--kernel-trace"]
        @test c.exec[10:17] == COUNTER_SETS[:amd][:occupancy].metrics
        @test c.exec[end-2:end] == ["--", "julia", "run.jl"]
        cmd = setenv(`julia run.jl`, ["TEST_KEY=1"]; dir = "/tmp")
        c = hw_counter_command(NsightCompute(), cmd; set = :fp64, dir = "out space", name = "probe",
            kernel = r"probe.*", launch_skip = 1, launch_count = 2, timeout_s = nothing)
        @test c.env == ["TEST_KEY=1"] && c.dir == "/tmp"
        @test "out space/probe_ncu.csv" in c.exec && "regex:probe.*" in c.exec
        @test c.exec[findfirst(==("--clock-control"), c.exec) + 1] == "none"
        @test c.exec[findfirst(==("--launch-count"), c.exec) + 1] == "2"
        @test "--print-metric-name" ∉ c.exec # unsupported with --page raw
        @test c.exec[findfirst(==("--target-processes"), c.exec) + 1] == "application-only"
        call = hw_counter_command(NsightCompute(), cmd; dir = "out", name = "p", target_processes = :all, timeout_s = nothing)
        @test call.exec[findfirst(==("--target-processes"), call.exec) + 1] == "all"
        @test hw_counter_export_command("p.ncu-rep"; output = "p.csv").exec ==
            ["ncu", "--import", "p.ncu-rep", "--csv", "--page", "raw", "--print-units", "base", "--log-file", "p.csv"]
        @test !hw_counters_available(GPUDiagnostics.KA.CPU())
        st = hw_counter_status(NsightCompute(); executable = "gpudiag_no_such_profiler")
        @test !st.available && ismissing(st.permitted) && ismissing(st.executable)
        @test hw_counters_available(RocprofV3()) == (Sys.which("rocprofv3") !== nothing)
        # A collector carrying a path (what the AMDGPU extension builds) uses it by default;
        # the executable keyword still wins.
        mktempdir() do d
            exe = joinpath(d, "rocprofv3")
            write(exe, "#!/bin/sh\n"); chmod(exe, 0o755)
            st = hw_counter_status(RocprofV3(exe))
            @test st.available && st.executable == exe
            @test hw_counter_command(RocprofV3(exe), `true`; dir = "o", name = "p", timeout_s = nothing).exec[3] == exe
            @test hw_counter_command(RocprofV3(exe), `true`; dir = "o", name = "p", timeout_s = nothing,
                executable = "/custom/profiler").exec[3] == "/custom/profiler"
            @test !hw_counter_status(RocprofV3(exe); executable = "gpudiag_no_such_profiler").available
        end
        @test RocprofV3() == RocprofV3(nothing)
        @test_throws BackendUnsupported hw_counter_command(GPUDiagnostics.KA.CPU(), `true`; dir = "o", name = "p")
        for opts in ((; set = :absent), (; metrics = String[]), (; timeout_s = 0),
                (; launch_skip = -1), (; launch_count = 0), (; clock_control = :invalid), (; kill_after_s = -1),
                (; target_processes = :children))
            @test_throws ArgumentError hw_counter_command(NsightCompute(), `true`; dir = "o", name = "p", opts...)
        end
        @test_throws ArgumentError hw_counter_command(RocprofV3(), `true`; dir = "o", name = "../p")
        err = try hw_counter_command(RocprofV3(), `true`; dir = "o", name = "p", launch_skip = 1); nothing catch e; e end
        @test err isa ArgumentError && occursin("RocprofV3 does not accept the keyword(s) launch_skip", err.msg) && occursin("kernel_trace", err.msg)
        err = try hw_counter_command(NsightCompute(), `true`; dir = "o", name = "p", kernel_trace = false, launch_count = 0); nothing catch e; e end
        @test err isa ArgumentError && occursin("NsightCompute does not accept the keyword(s) kernel_trace", err.msg)
        @test_throws MethodError hw_counter_command(:nvidia, `true`; dir = "o", name = "p")
        @test_throws MethodError hw_counter_status(:amd)
        amd = hw_counter_command(RocprofV3(), `true`; dir = "o", name = "p",
            kernel = r"probe.*", kernel_trace = false, metrics = ["SQ_WAVES:device=0"], timeout_s = nothing)
        @test amd.exec[3:end] == ["rocprofv3", "--kernel-include-regex", "probe.*", "--pmc",
            "SQ_WAVES:device=0", "--output-format", "csv", "-d", "o", "-o", "p", "--", "true"]
        @test all(p -> p.passes == 1, values(COUNTER_SETS[:amd]))
        for v in (:amd, :nvidia)   # the notes are the per-preset reading guide, not boilerplate
            notes = [COUNTER_SETS[v][k].notes for k in (:issue, :occupancy, :memory, :fp64, :l2)]
            @test all(n -> length(n) > 200, notes) && allunique(notes)
        end
        @test occursin("FOUR counters", COUNTER_SETS[:amd][:l2].notes) && occursin("per-WAVE", COUNTER_SETS[:amd][:issue].notes)
        @test occursin("EVENT counts", COUNTER_SETS[:amd][:memory].notes) && occursin("per-THREAD", COUNTER_SETS[:nvidia][:fp64].notes)
    end

    @testset "ncu sections, window, stall and limiter presets" begin
        exec(; kw...) = hw_counter_command(NsightCompute(), `julia run.jl`; dir = "o", name = "p", timeout_s = nothing, kw...).exec
        after(e, flag) = e[findfirst(==(flag), e) + 1]
        # the default preset is :issue
        @test after(exec(), "--metrics") == join(COUNTER_SETS[:nvidia][:issue].metrics, ',')
        # sections alone replace the preset; with `set` or `metrics` both are collected
        e = exec(; section_set = :detailed, sections = ["WarpStateStats", "SchedulerStats"])
        @test after(e, "--set") == "detailed" && "--metrics" ∉ e
        @test e[findall(==("--section"), e) .+ 1] == ["WarpStateStats", "SchedulerStats"]
        @test "--metrics" ∉ exec(; sections = ["WarpStateStats"]) && "--metrics" ∉ exec(; sections = ("WarpStateStats",))
        @test after(exec(; section_set = "full", set = :stalls), "--metrics") == join(COUNTER_SETS[:nvidia][:stalls].metrics, ',')
        @test after(exec(; section_set = :detailed, metrics = ["dram__bytes.sum"]), "--metrics") == "dram__bytes.sum"
        @test "--profile-from-start" ∉ exec()
        @test after(exec(; profile_from_start = false), "--profile-from-start") == "off"
        for bad in ((; section_set = "a b"), (; sections = [""]), (; sections = "WarpStateStats"), (; sections = ["a,b"]))
            @test_throws ArgumentError exec(; bad...)
        end
        err = try hw_counter_command(RocprofV3(), `true`; dir = "o", name = "p", sections = ["X"]); nothing catch e; e end
        @test err isa ArgumentError && occursin("RocprofV3 does not accept", err.msg)
        @test_throws ArgumentError hw_counter_command(RocprofV3(), `true`; dir = "o", name = "p", set = :stalls)
        @test "--set" ∉ exec(; section_set = nothing, sections = nothing)   # "not given": the default preset
        # presets: every stall reason, the limiter inputs, the measured pass counts
        @test length(NVIDIA_STALL_REASONS) == 17 && allunique(first.(NVIDIA_STALL_REASONS))
        st = COUNTER_SETS[:nvidia][:stalls]
        @test length(st.metrics) == 18 && all(m -> m == "gpu__time_duration.sum" || startswith(m, "smsp__average_warps_issue_stalled_"), st.metrics)
        @test COUNTER_SETS[:nvidia][:limiter].metrics ⊇ st.metrics && "sm__maximum_warps_per_active_cycle_pct" in COUNTER_SETS[:nvidia][:limiter].metrics
        @test "sm__cycles_elapsed.avg.per_second" in COUNTER_SETS[:nvidia][:limiter].metrics
        @test st.passes == 7 && COUNTER_SETS[:nvidia][:limiter].passes == 11 && COUNTER_SETS[:nvidia][:fp64].passes == 3
        @test occursin("long_scoreboard", st.notes) && occursin("clock_control", COUNTER_SETS[:nvidia][:limiter].notes)
        @test !haskey(COUNTER_SETS[:amd], :stalls)
        # profiler marker and detection
        withenv(PROFILER_ENV => nothing, "CUDA_INJECTION64_PATH" => nothing, "LD_PRELOAD" => nothing,
                "NV_TPS_LAUNCH_TOKEN" => nothing, "NV_NSIGHT_INJECTION_TRANSPORT_TYPE" => nothing) do
            @test profiler_attached() === nothing
            withenv(() -> (@test profiler_attached() == "ncu"), PROFILER_ENV => "ncu")
            withenv(() -> (@test profiler_attached() == "nsight"), "NV_NSIGHT_INJECTION_TRANSPORT_TYPE" => "2")
            withenv(() -> (@test profiler_attached() == "cuda-injection"), "CUDA_INJECTION64_PATH" => "/x/libinj.so")
            withenv(() -> (@test profiler_attached() == "rocprofv3"), "LD_PRELOAD" => "/opt/rocm/lib/librocprofiler-sdk-tool.so")
            p = GPUDiagnostics.PeakProbe(:Float64, 1.0e12, 4, 1024, 256, 100, 1.0e-3, 3, missing, [4], [1024], [1.0e12])
            @test p.under_profiler === nothing && !haskey(diagnostics_dict(p; prefix = "peak_probe_"), "peak_probe_under_profiler")
            q = GPUDiagnostics.PeakProbe(:Float64, 1.0e12, 4, 1024, 256, 100, 1.0e-3, 3, missing, [4], [1024], [1.0e12], "ncu")
            @test diagnostics_dict(q; prefix = "peak_probe_")["peak_probe_under_profiler"] == "ncu"
            @test occursin("UNDER ncu", sprint(show, q))
        end
    end

    @testset "profiling window: one range per device" begin
        b = WindowBackend()
        mark(x) = push!(b.log, x)
        closed() = all(w -> w.open == 0, values(GPUDiagnostics._WINDOWS))
        ranges() = filter(x -> x isa Tuple, b.log)
        # nested: the inner window leaves the range open
        @test hw_counter_window(b) do
            hw_counter_window(() -> mark(:inner), b)
            mark(:outer)
            :r
        end === :r
        @test b.log == [:sync, (:start, 1), :inner, :sync, :outer, :sync, (:stop, 1)] && closed()
        # an inner failure that the outer catches leaves it open; an outer failure closes it
        empty!(b.log)
        @test_throws ErrorException hw_counter_window(b) do
            try
                hw_counter_window(() -> error("inner"), b)
            catch
            end
            mark(:after)
            error("outer")
        end
        @test b.log == [:sync, (:start, 1), :after, (:stop, 1)] && closed()
        # concurrent on one device: open until the last one ends
        empty!(b.log)
        first_in, second_in, first_out = Base.Event(), Base.Event(), Base.Event()
        t1 = @async (hw_counter_window(() -> (notify(first_in); wait(second_in)), b); mark(:first_out); notify(first_out))
        t2 = @async (wait(first_in); hw_counter_window(() -> (notify(second_in); wait(first_out); mark(:late)), b))
        @test _finished(t1, t2)
        @test filter(!=(:sync), b.log) == [(:start, 1), :first_out, :late, (:stop, 1)] && closed()
        # concurrent on two devices: a range each
        empty!(b.log)
        ev = [Base.Event(), Base.Event()]
        ts = map(1:2) do i
            @async (gpu_device!(b, i); hw_counter_window(() -> (notify(ev[i]); wait(ev[3 - i])), b))
        end
        @test _finished(ts...)
        @test sort(ranges()) == [(:start, 1), (:start, 2), (:stop, 1), (:stop, 2)] && closed()
        # `f` selects another device: the range closes on the one it opened on, and the selection stays
        empty!(b.log)
        task = @async (hw_counter_window(() -> gpu_device!(b, 2), b); gpu_device(b))
        @test fetch(task) == 2 && ranges() == [(:start, 1), (:stop, 1)] && closed()
        # a range that does not open leaves no window behind
        empty!(b.log)
        b.fail[] = :start
        @test_throws ErrorException hw_counter_window(() -> mark(:ran), b)
        @test closed() && :ran ∉ b.log
        # nor does one that does not close
        empty!(b.log)
        b.fail[] = :stop
        @test_throws ErrorException hw_counter_window(() -> nothing, b)
        b.fail[] = :none
        @test closed()
        hw_counter_window(() -> nothing, b)
        @test ranges() == [(:start, 1), (:stop, 1), (:start, 1), (:stop, 1)]
        # two tasks open at once: one range
        empty!(b.log)
        both = [Base.Event(), Base.Event()]
        ts = map(i -> (@async hw_counter_window(() -> (notify(both[i]); wait(both[3 - i])), b)), 1:2)
        @test _finished(ts...) && ranges() == [(:start, 1), (:stop, 1)] && closed()
        # a range that is slow to open on one device does not hold up another device
        empty!(b.log)
        b.gates[1] = Base.Event()
        t1 = @async hw_counter_window(() -> nothing, b)
        t2 = @async (gpu_device!(b, 2); hw_counter_window(() -> notify(b.gates[1]), b))
        done = _finished(t1, t2)
        notify(b.gates[1]); delete!(b.gates, 1)   # release t1 if it is still waiting
        @test done && sort(ranges()) == [(:start, 1), (:start, 2), (:stop, 1), (:stop, 2)]
        # another backend type has its own range on the same device number
        empty!(b.log)
        hw_counter_window(() -> hw_counter_window(() -> nothing, OtherWindowBackend(b)), b)
        @test ranges() == [(:start, 1), (:start, 1), (:stop, 1), (:stop, 1)] && closed()
    end

    @testset "multi-kernel collection: per-kernel table and limiter" begin
        f = joinpath(@__DIR__, "fixtures", "ncu", "multikernel_ncu.csv")
        @test kernel_name_stem("gpu_resort_rank_kernel_(CompilerMetadata<A, B>, Int32)") == "resort_rank_kernel"
        @test kernel_name_stem("gpu__fma_chain_kernel_(x)") == "_fma_chain_kernel"
        @test kernel_name_stem("plain") == "plain" && kernel_name_stem("gpu_") == "gpu_"
        @test kernel_name_stem("gpu_foo_(x)") == "foo" && kernel_name_stem("gpu_foo__(x)") == "foo_"
        @test kernel_name_stem("ns::(anonymous namespace)::foo(int)") == "ns::(anonymous namespace)::foo"
        @test kernel_name_stem("ns::(anonymous namespace)::foo<(int)3>(int)") == "ns::(anonymous namespace)::foo<(int)3>"
        @test kernel_name_stem("void foo<bar<(bool)1>, 2>(float*)") == "void foo<bar<(bool)1>, 2>"
        @test kernel_name_stem("X::operator()(int)") == "X::operator()"
        for op in ("<", "<<", ">", ">>", "->", "<=", "<<=")
            @test kernel_name_stem("X::operator$op(int)") == "X::operator$op"
            @test kernel_name_stem("X::operator$op <(int)3>(int)") == "X::operator$op <(int)3>"   # demanglers put the space
        end
        @test kernel_name_stem("void apply_operator<(int)3>(float*)") == "void apply_operator<(int)3>"   # not an operator
        @test kernel_name_stem("apply_operator()") == "apply_operator" && kernel_name_stem("gpu_operator()") == "operator"
        @test kernel_name_stem("enable_if<(1<2), void>::type foo(int)") == "enable_if<(1<2), void>::type foo"
        @test kernel_name_stem("a>b(int)") == "a>b"
        @test kernel_name_stem("gpu_größe_kernel_(x)") == "größe_kernel"
        @test length(kernel_name_stem("a"^100_000 * "(x)")) == 100_000
        @test isempty(GPUDiagnostics._top_stalls(Dict("nvidia_stall_long_scoreboard" => 0.0), 3))
        @test_throws ArgumentError hw_counters(f)   # several kernels: hw_counters still wants one
        hs = hw_counter_kernels(f)
        @test length(hs) == 8 && sum(length, hs) == 21 && allunique(h.kernel for h in hs)
        @test all(h -> h.provenance["passes"] == 25, hs)
        @test length(hw_counter_kernels(f; kernel = r"^gpu_scan_tile")) == 5
        @test_throws ArgumentError hw_counter_kernels(f; kernel = "no_such_kernel")
        rank = only(hw_counter_kernels(f; kernel = "gpu_resort_rank_kernel_"))
        d = hw_counter_derived(rank)
        @test d["nvidia_sm_throughput"] ≈ [0.066] && d["nvidia_memory_throughput"] ≈ [0.2905]
        @test d["nvidia_stall_long_scoreboard"] ≈ [65.25] && d["nvidia_theoretical_occupancy"] ≈ [0.4167]
        lim = hw_counter_limiter(rank)
        @test lim.limiter === :latency_bound && lim.top_stall == "long_scoreboard" && occursin("SM 7%, memory 29%", lim.reason)
        @test hw_counter_limiter(rank; low = 0.05).limiter === :mixed
        @test hw_counter_limiter(rank; low = 0.3, bound = 0.3).limiter === :latency_bound
        @test hw_counter_limiter(rank; occupancy_low = 0.5).limiter === :occupancy_limited
        @test_throws ArgumentError hw_counter_limiter(rank; low = 0.7)

        t = hw_counter_by_kernel(f)
        @test [r.group for r in t] == ["resort_rank_kernel", "gather_all_kernel", "scan_tile_kernel", "tail_key_kernel"]
        @test [r.dispatches for r in t] == [1, 2, 17, 1] && length(t[3].kernels) == 5
        @test sum(r.share for r in t) ≈ 1
        @test t[1].share ≈ 0.6932 atol = 1e-4
        @test t[1].time_s ≈ 3.792384e-3
        @test [r.limiter for r in t] == [:latency_bound, :memory_bound, :latency_bound, :latency_bound]
        @test first(t[2].top_stalls) == ("lg_throttle" => t[2].nvidia_stall_lg_throttle)
        @test length(t[1].top_stalls) == 3 && all(p -> first(p) ∉ ("selected", "not_selected"), t[1].top_stalls)
        @test all(r -> keys(r) == keys(t[1]), t)
        @test hw_counter_limiter(t[2]) == (limiter = :memory_bound, dominant = :memory, fraction = t[2].fraction,
            top_stall = "lg_throttle", reason = t[2].limiter_reason)
        @test t[2].fraction == t[2].nvidia_memory_throughput && all(r -> r.dominant === :memory, t)
        # every label, from synthetic inputs (the fixture has no compute-bound or occupancy-limited kernel)
        lim(sm, mem, occ) = hw_counter_limiter((nvidia_sm_throughput = sm, nvidia_memory_throughput = mem, nvidia_active_occupancy = occ))
        @test lim(0.9, 0.2, 0.8).limiter === :compute_bound && lim(0.9, 0.2, 0.8).dominant === :compute
        @test lim(0.1, 0.9, 0.8).limiter === :memory_bound
        @test lim(0.1, 0.2, 0.1).limiter === :occupancy_limited && lim(0.1, 0.2, 0.8).limiter === :latency_bound
        @test lim(0.1, 0.59, 0.8).limiter === :mixed && lim(0.1, 0.59, 0.8).dominant === :memory && lim(0.1, 0.59, 0.8).fraction == 0.59
        @test ismissing(hw_counter_limiter((nvidia_sm_throughput = 0.5,)).dominant)
        @test occursin("occupancy unknown", hw_counter_limiter((nvidia_sm_throughput = 0.1, nvidia_memory_throughput = 0.2)).reason)
        @test ismissing(hw_counter_limiter((nvidia_sm_throughput = NaN, nvidia_memory_throughput = 0.5)).limiter)
        # the SM clock the collection ran at is in the reason: the label depends on it
        @test occursin("of peak at 2.0 of 2.66 GHz", hw_counter_limiter((nvidia_sm_throughput = 0.5, nvidia_memory_throughput = 0.5,
            nvidia_elapsed_clock_GHz = 1.998, nvidia_max_clock_GHz = 2.655)).reason)
        @test occursin("of peak at 2.0 GHz", hw_counter_limiter((nvidia_sm_throughput = 0.5, nvidia_memory_throughput = 0.5,
            nvidia_elapsed_clock_GHz = 1.998)).reason)
        mktempdir() do dir
            csv = joinpath(dir, "clock_ncu.csv")
            write(csv, "\"ID\",\"Kernel Name\",\"device__attribute_max_gpu_frequency_khz\",\"sm__cycles_elapsed.avg.per_second\"\n" *
                "\"\",\"\",\"\",\"hz\"\n\"0\",\"k(int)\",\"2655000\",\"1998000000\"\n")
            d = hw_counter_derived(only(hw_counter_kernels(csv)))
            @test d["nvidia_elapsed_clock_GHz"] ≈ [1.998] && d["nvidia_max_clock_GHz"] ≈ [2.655]
            write(csv, replace(read(csv, String), "2655000" => "0"))   # an unreported maximum is not 0 GHz
            @test !haskey(hw_counter_derived(only(hw_counter_kernels(csv))), "nvidia_max_clock_GHz")
        end
        # time weighting: the group mean lies between its kernels' values, weighted by duration
        scan = hw_counter_kernels(f; kernel = r"^gpu_scan_tile")
        v = [x for h in scan for x in hw_counter_derived(h)["nvidia_sm_throughput"]]
        w = [d.duration_s for h in scan for d in h.dispatches]
        @test t[3].nvidia_sm_throughput ≈ sum(v .* w) / sum(w)
        @test length(hw_counter_by_kernel(f; top_stalls = 1)[1].top_stalls) == 1
        @test [r.group for r in hw_counter_by_kernel(f; group = k -> "all")] == ["all"]
        @test hw_counter_by_kernel(f; limiter = (; bound = 0.8))[2].limiter === :mixed
        @test GPUDiagnostics.Tables.istable(t)
        # AMD: shares from real durations, no limiter rules
        a = hw_counter_by_kernel(joinpath(@__DIR__, "fixtures", "rocprof", "sq2_counter_collection.csv"))
        @test all(r -> ismissing(r.limiter) && occursin("NVIDIA only", r.limiter_reason), a)
        @test sum(r.share for r in a) ≈ 1
    end

    @testset "real NVIDIA FP64 collection" begin
        h = hw_counters(joinpath(@__DIR__, "fixtures", "ncu", "fp64_ncu.csv"); slots = 4096 * 32)
        @test length(h) == 2
        @test h["smsp__sass_thread_inst_executed_op_dfma_pred_on.sum"] == [131072, 131072]
        @test hw_counter_derived(h)["fp64_flop_per_slot"] == [2, 2]
        @test [d.duration_s for d in h.dispatches] ≈ [3.136e-6, 3.264e-6]
        @test first(h.dispatches).device["name"] == "NVIDIA test GPU"
        @test first(h.dispatches).device["device__attribute_confidential_computing_mode"] == "No-CC"
        @test all(d -> d.resources["grid_size"] == 4096 && d.resources["grid_blocks"] == 16, h.dispatches)
        # metadata columns of the raw page are not counters: launch attributes → resources, replay passes → provenance.
        # The fixture predates the pipe-fraction metric in the preset, so it is a subset of the preset.
        @test Set(h.counters) ⊆ Set(["gpu__time_duration.sum"; COUNTER_SETS[:nvidia][:fp64].metrics])
        @test "sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active" in COUNTER_SETS[:nvidia][:fp64].metrics
        @test !haskey(hw_counter_derived(h), "nvidia_fp64_pipe_peak_fraction")   # absent from this capture, not zero
        @test all(d -> d.resources["registers"] == 16 && d.resources["launch__registers_per_thread"] == 16 &&
            d.resources["launch__func_cache_config"] == "CachePreferNone" && d.resources["profiler__replayer_passes"] == 3, h.dispatches)
        @test h.provenance["passes"] == 3 && hw_counter_summary(h)["collection_passes"] == 3
        @test hw_counters(joinpath(@__DIR__, "fixtures", "ncu", "fp64_ncu.csv"); provenance = Dict("passes" => 1)).provenance["passes"] == 1
        s = hw_counter_summary(h)
        @test !any(k -> occursin("launch__", k) || occursin("profiler__", k) || occursin("device__", k), keys(s))
        @test s["resources_registers"] == 16 && s["device_name"] == "NVIDIA test GPU" && s["device_architecture"] == "12.0"
        @test first(h.dispatches).device["device__attribute_confidential_computing_mode"] == "No-CC"   # kept on the dispatch
        h.units["smsp__sass_thread_inst_executed_op_dfma_pred_on.sum"] = "Kinst"
        @test hw_counter_derived(h)["insts_per_slot_fp64_fma"] == [1000, 1000]
    end
end
