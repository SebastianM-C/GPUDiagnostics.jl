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
    @testset "MI300X v1.9 capture (idle, attribute table)" begin
        f = joinpath(@__DIR__, "fixtures", "amdgpu", "mi300x_gpu_metrics_v1_9_idle.bin")
        m = amd_gpu_metrics(f)
        @test m.version == v"1.9" && m.structure_size == 1150 && m.known
        @test m.hotspot_C == 42 && m.mem_temperature_C == 37 && m.socket_power_W == 194
        @test m.gfx_activity == 0 && m.umc_activity == 0
        @test m.gfxclk_MHz == [2098, 2100, 2110, 2105, 2103, 2109, 2105, 2115] && length(m.gfxclk_MHz) == 8
        @test m.accumulation_counter == 54536484 && m.ppt_residency_acc == 633093
        @test m.socket_thm_residency_acc == 0 && m.vr_thm_residency_acc == 0 && m.hbm_thm_residency_acc == 0 && m.prochot_residency_acc == 0
        @test ismissing(m.throttle_status) && ismissing(m.indep_throttle_status)
        @test length(m.attrs) == 47 && m.attrs[:current_gfxclk] isa Vector{UInt16}
        @test m.attrs[:gfx_busy_inst] == [1, 0, 0, 0, 0, 0, 0, 0] && length(m.attrs[:gfx_below_host_limit_ppt_acc]) == 8
        @test m.attrs[:gfx_below_host_limit_ppt_acc] isa Vector{UInt64} && m.attrs[:mem_max_bandwidth] == [5325]
        @test isequal(amd_gpu_metrics(read(f)), m)
        cols = GPUDiagnostics._gpu_metrics_columns(f)
        @test cols.throttle_acc_counter == 54536484 && cols.power_throttle_acc == 633093
        @test cols.xcd_clock_min_MHz == 2098 && cols.xcd_clock_max_MHz == 2115 && isnan(cols.amd_throttle_status)
        @test cols.thermal_throttle_acc == 0 && cols.prochot_acc == 0
        # fixed-layout revisions carry an empty attrs
        @test isempty(amd_gpu_metrics(joinpath(@__DIR__, "fixtures", "amdgpu", "w7900_gpu_metrics_idle.bin")).attrs)
        # truncated mid-table: what was parsed before the cut, missing after it, never a throw
        b = read(f)
        t = amd_gpu_metrics(b[1:120])   # entries 0–8 fit; the accumulation counter (entry 9) is cut
        @test t.version == v"1.9" && t.known && t.hotspot_C == 42 && t.socket_power_W == 194
        @test ismissing(t.accumulation_counter) && isempty(t.gfxclk_MHz) && length(t.attrs) < 47
        t8 = amd_gpu_metrics(b[1:8])   # header + count only
        @test t8.known && isempty(t8.attrs) && ismissing(t8.hotspot_C)
        # an unknown type index ends the walk without an error
        bad = copy(b); bad[9 + 2] |= 0x80   # type nibble → 8+
        @test isempty(amd_gpu_metrics(bad).attrs)
        # an unknown id is kept under a generic name
        odd = copy(b); odd[9 + 1] = 0xfc; odd[9 + 2] = (odd[9 + 2] & 0xf0) | 0x0f   # id bits 10..19 → 1023
        @test haskey(amd_gpu_metrics(odd).attrs, :attr_1023)
    end
    @testset "MI300X v1.9 under load: residency fractions through the stats path" begin
        fa = joinpath(@__DIR__, "fixtures", "amdgpu", "mi300x_gpu_metrics_v1_9_load_a.bin")
        fb = joinpath(@__DIR__, "fixtures", "amdgpu", "mi300x_gpu_metrics_v1_9_load_b.bin")
        a, b = amd_gpu_metrics(fa), amd_gpu_metrics(fb)
        @test a.version == v"1.9" && a.known && b.known
        @test a.accumulation_counter == 55257356 && a.ppt_residency_acc == 749905 && a.socket_thm_residency_acc == 0 && a.hbm_thm_residency_acc == 0
        @test b.accumulation_counter == 55262364 && b.ppt_residency_acc == 754247
        @test a.gfxclk_MHz == [1304, 1275, 1317, 1283, 1298, 1281, 1309, 1287] && a.hotspot_C == 59 && a.socket_power_W == 705
        @test b.gfxclk_MHz == [1292, 1262, 1306, 1272, 1289, 1270, 1296, 1272] && b.hotspot_C == 68 && b.socket_power_W == 707
        @test a.gfx_activity == 1.0
        ca, cb = GPUDiagnostics._gpu_metrics_columns(fa), GPUDiagnostics._gpu_metrics_columns(fb)
        @test ca.xcd_clock_min_MHz == 1275 && ca.xcd_clock_max_MHz == 1317
        # two busy rows of one device, exactly as the sampler child would have written them
        cols = vcat([:t_rel_s, :device, :compute_util], collect(keys(ca)))
        rows = [[0.0, 1.0, a.gfx_activity, values(ca)...], [5.0, 1.0, b.gfx_activity, values(cb)...]]
        tel = GPUTelemetry(cols, permutedims(hcat(rows...)), 2, 5.0, 5.0, 0.0, false, nothing, :none)
        st = gpu_telemetry_stats(tel)
        @test isapprox(st["power_violation_fraction"], 4342 / 5008; atol = 1e-3)   # ≈ 0.867
        @test st["thermal_violation_fraction"] == 0 && st["hbm_thermal_violation_fraction"] == 0
        @test st["vr_thermal_violation_fraction"] == 0 && st["prochot_fraction"] == 0
        @test st["xcd_clock_min_MHz_busy_median"] == (1275 + 1262) / 2
    end
    @testset "MI300X v1.6, AMD DKMS variant (1664 bytes, idle)" begin
        f = joinpath(@__DIR__, "fixtures", "amdgpu", "mi300x_gpu_metrics_v1_6_dkms_idle.bin")
        m = amd_gpu_metrics(f)
        @test m.version == v"1.6" && m.structure_size == 1664 && m.known
        @test m.hotspot_C == 42 && m.mem_temperature_C == 40 && m.socket_power_W == 120
        @test m.gfx_activity == 0.01 && m.umc_activity == 0
        @test m.gfxclk_MHz == [160, 160, 160, 160, 159, 159, 158, 159] && length(m.gfxclk_MHz) == 8
        @test m.accumulation_counter == 10657785 && m.prochot_residency_acc == 0 && m.vr_thm_residency_acc == 0
        # the firmware quirk on this driver: three residencies read as the whole accumulation counter at idle
        @test m.ppt_residency_acc == m.socket_thm_residency_acc == m.hbm_thm_residency_acc == 10657785
        @test ismissing(m.throttle_status) && isempty(m.attrs)
        cols = GPUDiagnostics._gpu_metrics_columns(f)
        @test cols.throttle_acc_counter == 10657785 && cols.power_throttle_acc == 10657785
        @test cols.xcd_clock_min_MHz == 158 && cols.xcd_clock_max_MHz == 160
        L = GPUDiagnostics._GPU_METRICS_LAYOUTS
        @test L[(1, 6, 1664)] == L[(1, 6, 312)]   # the vendor variant shares every offset with upstream
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
        @test L == GPUDiagnostics._GPU_METRICS_LAYOUTS && length(L) >= 21
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
