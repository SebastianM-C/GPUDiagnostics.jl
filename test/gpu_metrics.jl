using Test, GPUDiagnostics

# Build a gpu_metrics blob of a given revision from the layout table itself, so the decoder can be
# exercised on revisions no card here produces (MI300's v1.8 residencies).
function synth_gpu_metrics(fmt, con, size; fields...)
    bytes = zeros(UInt8, size)
    bytes[1:2] = reinterpret(UInt8, [UInt16(size)]); bytes[3] = fmt; bytes[4] = con
    layout = GPUDiagnostics._GPU_METRICS_LAYOUTS[(fmt, con, size)]
    for (k, v) in pairs(fields)
        off, w, n = layout[k]
        T = w == 2 ? UInt16 : w == 4 ? UInt32 : UInt64
        vals = v isa AbstractVector ? v : [v]
        for (i, x) in enumerate(vals)
            bytes[(off + 1 + w * (i - 1)):(off + w * i)] = reinterpret(UInt8, [T(x)])
        end
    end
    return bytes
end

@testset "amdgpu gpu_metrics" begin
    @testset "W7900 v1.3 capture (idle)" begin
        f = joinpath(@__DIR__, "fixtures", "amdgpu", "w7900_gpu_metrics_idle.bin")
        m = amd_gpu_metrics(f)
        @test m.version == v"1.3" && m.structure_size == 120 && m.known
        @test m.gfx_activity == 0 && m.umc_activity == 0.01 && m.socket_power_W == 9
        @test m.gfxclk_MHz == [3.0]                         # single die, idle clock
        @test m.throttle_status == 2 && m.indep_throttle_status == UInt64(1) << 36
        @test amd_throttle_reasons(m.indep_throttle_status) == [:temp_hotspot]   # the RDNA3 idle quirk, recorded
        @test all(ismissing, (m.accumulation_counter, m.ppt_residency_acc, m.socket_thm_residency_acc))
        @test isequal(amd_gpu_metrics(read(f)), m)
        cols = GPUDiagnostics._gpu_metrics_columns(f)
        @test cols.amd_throttle_status == 2.0^36 && cols.xcd_clock_min_MHz == 3 && cols.xcd_clock_max_MHz == 3
        @test isnan(cols.throttle_acc_counter) && isnan(cols.power_throttle_acc)
    end
    @testset "MI300-class v1.8: residencies and per-XCD clocks" begin
        b = synth_gpu_metrics(1, 8, 344; gfx_activity = 98, umc_activity = 42, socket_power_W = 745,   # the SMU reports percent
            accumulation_counter = 1_000_000, ppt_residency_acc = 800_000, socket_thm_residency_acc = 0,
            vr_thm_residency_acc = 0, hbm_thm_residency_acc = 12, prochot_residency_acc = 0,
            gfxclk_MHz = [1250, 1300, 1250, 1275, 1250, 1300, 1250, 1250])
        m = amd_gpu_metrics(b)
        @test m.version == v"1.8" && m.known && m.gfx_activity == 0.98 && m.socket_power_W == 745
        @test m.gfxclk_MHz == [1250, 1300, 1250, 1275, 1250, 1300, 1250, 1250]
        @test m.ppt_residency_acc == 800_000 && m.accumulation_counter == 1_000_000 && m.hbm_thm_residency_acc == 12
        @test ismissing(m.throttle_status) && ismissing(m.indep_throttle_status)   # v1.6+ has residencies instead
        cols = GPUDiagnostics._gpu_metrics_columns(b)
        @test isnan(cols.amd_throttle_status) && cols.xcd_clock_min_MHz == 1250 && cols.xcd_clock_max_MHz == 1300
        @test cols.power_throttle_acc == 800_000 && cols.throttle_acc_counter == 1_000_000
        # unpopulated XCD slots (0 or all-ones) drop out; all-ones sensors are missing
        b2 = synth_gpu_metrics(1, 8, 344; gfxclk_MHz = [1250, 0, 0xffff, 0, 0, 0, 0, 0], socket_power_W = 0xffff)
        @test amd_gpu_metrics(b2).gfxclk_MHz == [1250] && ismissing(amd_gpu_metrics(b2).socket_power_W)
        # v1.4 has the per-XCD clocks and a raw SMU mask but no independent remap: the raw mask is what travels
        b3 = synth_gpu_metrics(1, 4, 288; throttle_status = 0x5, gfxclk_MHz = fill(2100, 8))
        @test GPUDiagnostics._gpu_metrics_columns(b3).amd_throttle_status == 5
    end
    @testset "unknown revisions and bad input" begin
        m = amd_gpu_metrics(UInt8[0x14, 0x00, 0x09, 0x09, zeros(UInt8, 16)...])
        @test m.version == v"9.9" && !m.known && ismissing(m.throttle_status) && isempty(m.gfxclk_MHz)
        m = amd_gpu_metrics(UInt8[0x64, 0x00, 0x01, 0x03, zeros(UInt8, 96)...])   # v1.3 header, wrong size
        @test !m.known && ismissing(m.indep_throttle_status)
        @test_throws ArgumentError amd_gpu_metrics(UInt8[1, 2])
        @test all(isnan, values(GPUDiagnostics._gpu_metrics_columns("/nonexistent/gpu_metrics")))
        @test all(isnan, values(GPUDiagnostics._gpu_metrics_columns(nothing)))
    end
    @testset "layout data file" begin
        L = GPUDiagnostics._load_gpu_metrics_layouts(GPUDiagnostics._GPU_METRICS_LAYOUTS_FILE)
        @test L == GPUDiagnostics._GPU_METRICS_LAYOUTS && length(L) >= 20
        @test L[(1, 3, 120)][:indep_throttle_status] == (112, 8, 1) && L[(1, 3, 120)][:gfxclk_MHz] == (54, 2, 1)
        @test L[(1, 8, 344)][:ppt_residency_acc] == (48, 4, 1) && L[(1, 8, 344)][:gfxclk_MHz] == (296, 2, 8)
        @test all(v -> all(x -> x[1] >= 4 && x[2] in (1, 2, 4, 8) && x[3] >= 1, values(v)), values(L))
    end
    @testset "throttler bit names" begin
        @test amd_throttle_reasons(0) == Symbol[] && amd_throttle_reasons(NaN) == Symbol[]
        @test amd_throttle_reasons(1) == [:ppt0] && amd_throttle_reasons(2.0^36) == [:temp_hotspot]
        @test amd_throttle_reasons(2.0^0 + 2.0^16 + 2.0^36) == [:ppt0, :tdc_gfx, :temp_hotspot]
        @test amd_throttle_reasons(2.0^57) == [:fit] && amd_throttle_reasons(2.0^10) == [:bit_10]
        @test_throws ArgumentError amd_throttle_reasons(-1)
        @test length(AMD_THROTTLER_BITS) == 34
    end
    @testset "sysfs source with the blob, and the stats" begin
        d = mktempdir()
        write(joinpath(d, "p"), "9000000\n"); write(joinpath(d, "b"), "0\n"); write(joinpath(d, "v"), "1\n")
        cp(joinpath(@__DIR__, "fixtures", "amdgpu", "w7900_gpu_metrics_idle.bin"), joinpath(d, "gpu_metrics"))
        src = sampler_source("sysfs:1:$d/p:$d/b:-:$d/v:-:-:-:-:-:$d/gpu_metrics", :none)
        nt = GPUDiagnostics.sample!(src)
        @test nt.amd_throttle_status == 2.0^36 && nt.xcd_clock_max_MHz == 3 && isnan(nt.throttle_reasons)
        @test isnan(nt.power_throttle_acc) && haskey(nt, :prochot_acc)
        # without the 11th field the column set is the 0.2 one
        @test !haskey(GPUDiagnostics.sample!(sampler_source("sysfs:1:$d/p:$d/b:-:$d/v", :none)), :amd_throttle_status)
        @test_throws ArgumentError sampler_source("sysfs:1:$d/p:$d/b:-:$d/v:-:-:-:-:-:-:-", :none)
        # the MI300-like synthetic device through the real child: residency fractions and the bitmask fractions
        # (the CPU backend gets synthetic sources for the duration of this testset, as the sampler tests do)
        CPU = GPUDiagnostics.KA.CPU
        GPUDiagnostics.gpu_sampler_sources(::CPU, ids::AbstractVector{<:Integer}, counters::Symbol) =
            (specs = ["synthetic:$i" for i in ids], packages = Base.PkgId[])
        r, telem = with_gpu_sampler(GPUDiagnostics.KA.CPU(), 0.1; devices = 3:3) do
            sleep(2.5); :ok
        end
        @test r == :ok && telem.ticks >= 5 && haskey(telem, :power_throttle_acc)
        @test all(==(1.0), telem[:amd_throttle_status]) && all(isnan, telem[:throttle_reasons])
        st = gpu_telemetry_stats(telem)
        @test st["power_violation_fraction"] ≈ 0.8 && st["thermal_violation_fraction"] == 0
        @test st["hbm_thermal_violation_fraction"] == 0 && st["prochot_fraction"] == 0
        @test st["power_throttled_fraction"] == 1 && st["thermal_throttled_fraction"] == 0   # bit 0 = :ppt0
        @test st["xcd_clock_max_MHz_busy_median"] == 1300 && st["xcd_clock_min_MHz_mean"] == 1250
        # NVML-side bitmask fractions: synthetic device 1 reports sw_power_cap | sw_thermal_slowdown while busy
        _, t12 = with_gpu_sampler(GPUDiagnostics.KA.CPU(), 0.1; devices = 1:2) do
            sleep(1.5); nothing
        end
        s12 = gpu_telemetry_stats(t12)
        @test s12["power_throttled_fraction"] == 1 && s12["thermal_throttled_fraction"] == 1
        @test !haskey(s12, "power_violation_fraction")   # no residency columns on those devices
        Base.delete_method(only(methods(GPUDiagnostics.gpu_sampler_sources, (CPU, AbstractVector{<:Integer}, Symbol))))
    end
end
