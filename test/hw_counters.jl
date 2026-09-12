using Test, GPUDiagnostics
import Statistics

struct CounterTestBackend{C <: HWCounterCollector} <: GPUDiagnostics.KA.GPU end
GPUDiagnostics.supports(::CounterTestBackend, ::Val{:hw_counters}) = true
GPUDiagnostics.backend_counter_collector(::CounterTestBackend{C}) where {C} = C()

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
    override[devid]["architecture"] = "gfx1100"
    @test all(ismissing, hw_counter_derived(hw_counters(fx; name = "sq1", device_overrides = override))["amd_resident_waves"])

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
            @test first(c.exec) == tool
            @test c.exec == hw_counter_command(collector, cmd; dir = "out", name = "probe", timeout_s = nothing).exec
            @test c.env == cmd.env && c.dir == cmd.dir
            st = hw_counter_status(backend; executable = "gpudiag_no_such_profiler")
            @test st.tool == Symbol(tool) && !st.available && ismissing(st.permitted)
            @test !hw_counters_available(backend; executable = "gpudiag_no_such_profiler")
            @test first(hw_counter_command(backend, cmd; dir = "out", name = "probe",
                executable = "/custom/profiler", timeout_s = nothing).exec) == "/custom/profiler"
        end
        c = hw_counter_command(RocprofV3(), `julia run.jl`; set = :occupancy, dir = "out", name = "probe")
        @test c.exec[1:6] == ["timeout", "-k", "20", "600", "rocprofv3", "--kernel-trace"]
        @test c.exec[8:15] == COUNTER_SETS[:amd][:occupancy].metrics
        @test c.exec[end-2:end] == ["--", "julia", "run.jl"]
        cmd = setenv(`julia run.jl`, ["TEST_KEY=1"]; dir = "/tmp")
        c = hw_counter_command(NsightCompute(), cmd; set = :fp64, dir = "out space", name = "probe",
            kernel = r"probe.*", launch_skip = 1, launch_count = 2, timeout_s = nothing)
        @test c.env == ["TEST_KEY=1"] && c.dir == "/tmp"
        @test "out space/probe_ncu.csv" in c.exec && "regex:probe.*" in c.exec
        @test c.exec[findfirst(==("--clock-control"), c.exec) + 1] == "none"
        @test c.exec[findfirst(==("--launch-count"), c.exec) + 1] == "2"
        @test "--print-metric-name" ∉ c.exec # unsupported with --page raw
        @test hw_counter_export_command("p.ncu-rep"; output = "p.csv").exec ==
            ["ncu", "--import", "p.ncu-rep", "--csv", "--page", "raw", "--print-units", "base", "--log-file", "p.csv"]
        @test !hw_counters_available(GPUDiagnostics.KA.CPU())
        st = hw_counter_status(NsightCompute(); executable = "gpudiag_no_such_profiler")
        @test !st.available && ismissing(st.permitted) && ismissing(st.executable)
        @test hw_counters_available(RocprofV3()) == (Sys.which("rocprofv3") !== nothing)
        @test_throws BackendUnsupported hw_counter_command(GPUDiagnostics.KA.CPU(), `true`; dir = "o", name = "p")
        for opts in ((; set = :absent), (; metrics = String[]), (; timeout_s = 0),
                (; launch_skip = -1), (; launch_count = 0), (; clock_control = :invalid), (; kill_after_s = -1))
            @test_throws ArgumentError hw_counter_command(NsightCompute(), `true`; dir = "o", name = "p", opts...)
        end
        @test_throws ArgumentError hw_counter_command(RocprofV3(), `true`; dir = "o", name = "../p")
        @test_throws MethodError hw_counter_command(RocprofV3(), `true`; dir = "o", name = "p", launch_skip = 1)
        @test_throws MethodError hw_counter_command(NsightCompute(), `true`; dir = "o", name = "p", kernel_trace = false)
        @test_throws MethodError hw_counter_command(:nvidia, `true`; dir = "o", name = "p")
        @test_throws MethodError hw_counter_status(:amd)
        amd = hw_counter_command(RocprofV3(), `true`; dir = "o", name = "p",
            kernel = r"probe.*", kernel_trace = false, metrics = ["SQ_WAVES:device=0"], timeout_s = nothing)
        @test amd.exec == ["rocprofv3", "--kernel-include-regex", "probe.*", "--pmc",
            "SQ_WAVES:device=0", "--output-format", "csv", "-d", "o", "-o", "p", "--", "true"]
        @test all(p -> p.passes == 1, values(COUNTER_SETS[:amd]))
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
        h.units["smsp__sass_thread_inst_executed_op_dfma_pred_on.sum"] = "Kinst"
        @test hw_counter_derived(h)["insts_per_slot_fp64_fma"] == [1000, 1000]
    end
end
