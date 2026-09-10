# Backend conformance: every feature a backend declares through `supports` has working entry
# points of the documented shape, and every feature it does not declare fails with
# `BackendUnsupported` — never a `MethodError`, never a vendor-named string. Runs on the CPU
# backend and on a vendor-less backend in the regular tests; the same function runs against a
# real vendor backend (with a compiled kernel of that backend passed as `kernel`) in GPU CI.

using GPUDiagnostics, Test
using GPUDiagnostics: FEATURES

"""
    conformance(backend; kernel = nothing, peak_threads = 2^12)

Check `backend` against its declared capabilities. `kernel` is a `CompiledKernel` of this
backend for the `:resources` / `:native_mix` / `:ir_mix` shape checks (skipped when `nothing`;
the *unsupported* checks run regardless with a placeholder kernel). `peak_threads` sizes the
FP64 probe.
"""
function conformance(backend; kernel = nothing, peak_threads = 2^12)
    caps = capabilities(backend)
    placeholder = CompiledKernel("k", "sig", 256, nothing)
    has(f) = f in caps

    @testset "conformance: $(nameof(typeof(backend)))" begin
        @test caps ⊆ collect(FEATURES)
        @test all(f -> supports(backend, f) == supports(backend, Val(f)), FEATURES)
        @test all(f -> has(f) == supports(backend, f), FEATURES)

        @testset ":devices" begin
            if has(:devices)
                n = gpu_device_count(backend)
                @test n isa Int && n ≥ 1
                d = gpu_device(backend)
                @test d isa Int && 1 ≤ d ≤ n
                @test gpu_device!(backend, d) == d
                @test gpu_name(backend) isa AbstractString
                @test gpu_arch(backend) isa AbstractString
            else
                for f in (gpu_device_count, gpu_device, gpu_name, gpu_arch)
                    @test_throws BackendUnsupported f(backend)
                end
                @test_throws BackendUnsupported gpu_device!(backend, 1)
            end
        end

        @testset ":device_props" begin
            if has(:device_props)
                @test gpu_sm_count(backend) isa Union{Missing, Int}
                @test gpu_max_threads_per_sm(backend) isa Union{Missing, Int}
                m = gpu_memory_info(backend)
                @test all(k -> hasproperty(m, k), (:total, :free, :used))
                @test thread_fill_occupancy(backend, 1024) isa Union{Missing, Float64}
            else
                for f in (gpu_sm_count, gpu_max_threads_per_sm, gpu_memory_info)
                    @test_throws BackendUnsupported f(backend)
                end
                @test_throws BackendUnsupported thread_fill_occupancy(backend, 1024)
            end
        end

        @testset ":events" begin
            if has(:events)
                e0 = gpu_event(backend)
                e1 = gpu_event(backend)
                @test gpu_elapsed(e0, e1) isa Float64 && gpu_elapsed(e0, e1) ≥ 0
                timer = LaunchTimer()
                lane = launch_lane(timer, backend)
                t0 = launch_tick(timer, backend)
                launch_tock!(timer, lane, backend, t0)
                lt = launch_times(timer)
                @test length(lt) == 1 && length(only(values(lt))) == 1
            else
                @test_throws BackendUnsupported gpu_event(backend)
            end
        end

        @testset ":telemetry" begin
            if has(:telemetry)
                @test gpu_power(backend) isa Union{Missing, Float64}
                u = gpu_utilization(backend)
                @test hasproperty(u, :compute) && hasproperty(u, :memory)
                s = gpu_sample(backend, gpu_device(backend))
                @test all(k -> hasproperty(s, k), (:power_W, :compute_util, :mem_util, :vram_used_B))
                src = gpu_sampler_sources(backend, [gpu_device(backend)], :none)
                @test hasproperty(src, :specs) && hasproperty(src, :packages) && length(src.specs) == 1
            else
                @test_throws BackendUnsupported gpu_power(backend)
                @test_throws BackendUnsupported gpu_utilization(backend)
                @test_throws BackendUnsupported gpu_sampler_sources(backend, [1], :auto)
                @test_throws BackendUnsupported gpu_sample(backend, 1)
            end
            has(:telemetry_counters) && @test has(:telemetry)   # counters ride in the sample
        end

        @testset ":peak_flops / :fp64" begin
            if has(:peak_flops)
                p = measure_peak_flops(backend, Float32; n_threads = peak_threads, trials = 1, target_seconds = 0.01)
                @test p isa Float64 && p > 0
                if has(:fp64)
                    p64 = measure_peak_flops(backend, Float64; n_threads = peak_threads, trials = 1, target_seconds = 0.01)
                    @test p64 isa Float64 && p64 > 0
                else
                    @test_throws BackendUnsupported measure_peak_flops(backend, Float64)
                end
            else
                @test_throws BackendUnsupported measure_peak_flops(backend)
            end
        end

        @testset ":kernel_inventory" begin
            if has(:kernel_inventory)
                ks = compiled_kernels(backend)
                @test ks isa Vector{<:CompiledKernel}
                @test compiled_kernels(backend; pattern = r"no such kernel \d{9}") |> isempty
            else
                @test_throws BackendUnsupported compiled_kernels(backend)
            end
        end

        @testset ":resources / :occupancy" begin
            if has(:resources)
                has(:occupancy) && @test has(:resources)
                if kernel !== nothing
                    r = kernel_resources(backend, kernel)
                    @test all(k -> hasproperty(r, k), (:name, :registers, :local_mem_bytes, :shared_mem_bytes))
                end
            else
                @test !has(:occupancy)
                @test_throws BackendUnsupported kernel_resources(backend, placeholder)
            end
        end

        @testset ":native_mix" begin
            if has(:native_mix)
                if kernel !== nothing
                    m = kernel_instruction_mix(backend, kernel; ir = false)
                    @test hasproperty(m, :counts) && hasproperty(m, :hot_loop) && m.coverage isa Float64
                end
            else
                @test_throws BackendUnsupported kernel_instruction_mix(backend, placeholder)
            end
        end

        @testset ":ir_mix" begin
            if has(:ir_mix)
                if kernel !== nothing
                    m = kernel_ir_mix(backend, kernel)
                    @test hasproperty(m, :counts) && m.total isa Int
                end
            else
                @test_throws BackendUnsupported kernel_ir_mix(backend, placeholder)
            end
        end
    end
    return caps
end
