# CUDA-only section of runtests.jl (uses its `backend`): the ISA report under a register cap.
# 24 Float64 values live across the loop need far more than 32 registers, so a cap of 32 binds:
# the report must show the capped, spilling binary, not the uncapped one (issue #41).
using Base.Cartesian: @nexprs

function gpudiag_capped_kernel(out, x, n)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    @inbounds if i <= length(out)
        @nexprs 24 k -> v_k = x[i] * (0.1 * k)
        for _ in Int32(1):n
            w = v_1
            @nexprs 23 k -> v_k = fma(v_k, v_{k + 1}, 0.5)
            v_24 = fma(v_24, w, 0.5)
        end
        s = 0.0
        @nexprs 24 k -> s += v_k
        out[i] = s
    end
    return
end

@testset "register cap: the ISA report describes the capped binary" begin
    xs = CUDA.rand(Float64, 4096); outs = similar(xs)
    launch_free() = (@cuda threads = 256 blocks = 16 gpudiag_capped_kernel(outs, xs, Int32(10)); CUDA.synchronize())
    launch_capped() = (@cuda threads = 256 blocks = 16 maxregs = 32 gpudiag_capped_kernel(outs, xs, Int32(10)); CUDA.synchronize())
    free, capped = map((launch_free, launch_capped)) do launch
        kernel_resources(backend, only(compiled_kernels(launch, backend; pattern = r"gpudiag_capped_kernel")))
    end
    @info "register cap" free = free.registers capped = capped.registers spill = capped.isa["spill_store_bytes"]
    @test free.registers > 32 && free.isa["spill_store_bytes"] == 0
    @test capped.registers <= 32 && capped.isa["spill_store_bytes"] > 0
    for r in (free, capped)
        @test r.isa["ptxas_registers"] == r.registers && r.isa["ptxas_matches_attributes"] === true
    end
end
