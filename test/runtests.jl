using SpacetimeSpectralPassExamples
using SparseArrays
using StaticArrays
using Test

grid = range(0, 1, 21)
maxerr(c, exact, t) = maximum(abs(evalspace(c, x) - exact(t, x)) for x in grid)

@testset "SpacetimeSpectralPassExamples" begin
    @testset "block systems assemble sparsely" begin
        @test heat_problem_rect(1.0, 20, 16).A isa SparseMatrixCSC
        @test wave_problem_rect(1.0, 20, 16).A isa SparseMatrixCSC
        @test wave_problem_rect(1.0, 20, 16; m2 = 20.0).A isa SparseMatrixCSC
    end

    @testset "heat peel-off equals re-expansion of the final slice" begin
        Nt, Nx, Δt = 24, 16, 0.3
        g(t) = 1 + 0.25sin(t)
        forcing_time(t) = π^2 * g(t) + 0.25cos(t)
        prob = heat_problem_rect(Δt, Nt, Nx)
        rhs = heat_rhs_rect(prob, 0.0, Δt; forcing_time = forcing_time)
        U = heatblock_rect(prob, g(0.0) .* spacecoeffs(sinpi, Nx), rhs)
        @test peel(U) ≈ spacecoeffs(x -> evalblock_final(U, x), Nx) atol = 1e-10
    end

    @testset "heat block matches manufactured solution, global and blocked" begin
        Nt, Nx, T = 24, 24, 10.0
        g(t) = 1 + 0.25sin(t)
        forcing_time(t) = π^2 * g(t) + 0.25cos(t)
        exact(t, x) = g(t) * sinpi(x)
        for L in (1, 4)
            Δt = T / L
            prob = heat_problem_rect(Δt, Nt, Nx)
            rhsblocks = heat_rhs_blocks(prob, Δt, L; forcing_time = forcing_time)
            cT = stepsolve_heat_rect(g(0.0) .* spacecoeffs(sinpi, Nx), Δt, L, Nt, Nx; rhsblocks = rhsblocks)
            @test maxerr(cT, exact, T) < 1e-9
        end
    end

    @testset "wave first-order system and scalar derivative peeling reach t=10" begin
        Nt, Nx, T, mode = 48, 32, 10.0, 2
        exact(t, x) = cospi(mode * t) * sinpi(mode * x)
        u0 = spacecoeffs(x -> sinpi(mode * x), Nx)
        us, _ = stepsolve_wave_rect(u0, zeros(Nx), T / 2, 2, Nt, Nx)
        @test maxerr(us, exact, T) < 1e-9
        ud, _ = stepsolve_wave_scalar_rect(u0, zeros(Nx), T / 2, 2, Nt, Nx)
        @test maxerr(ud, exact, T) < 1e-9
    end

    @testset "velocity peel-off matches analytic u_t" begin
        Nt, Nx, Δt = 24, 20, 0.5
        u0 = spacecoeffs(sinpi, Nx)
        _, v = stepsolve_wave_scalar_rect(u0, zeros(Nx), Δt, 1, Nt, Nx)
        vt(t, x) = -π * sin(π * Δt) * sinpi(x)
        @test maxerr(v, vt, Δt) < 1e-8
    end

    @testset "Klein-Gordon first-order system reaches t=10" begin
        Nt, Nx, T, mode, m2 = 48, 32, 10.0, 2, 20.0
        ω = sqrt((mode * π)^2 + m2)
        exact(t, x) = cos(ω * t) * sinpi(mode * x)
        u0 = spacecoeffs(x -> sinpi(mode * x), Nx)
        u, _ = stepsolve_wave_rect(u0, zeros(Nx), T / 4, 4, Nt, Nx; m2 = m2)
        @test maxerr(u, exact, T) < 1e-9
    end

    @testset "disk fractional heat block reproduces manufactured solution" begin
        for beta in (1.0, 0.5)
            nblocks, Nt, Δt = 3, 5, 0.25
            K = zernike_ncoeffs(nblocks)
            phi = zeros(K); phi[1] = 1.0; phi[2] = -0.1; phi[4] = 0.15
            g(t) = 1 + t                       # linear in time, exactly representable
            C, L = disk_operators_cached(nblocks; beta = beta)
            prob = disk_problem_sparse(beta, Nt, Δt, nblocks)
            Fdt = timecoeffs_basis(U32, _ -> 1.0, 0.0, Δt, Nt)
            Ft  = timecoeffs_basis(U32, g, 0.0, Δt, Nt)
            rhs = vec(Fdt * transpose(C * phi) + Ft * transpose(L * phi))[prob.keep]
            U = disk_solve_from_sparse(prob, g(0.0) .* phi, [rhs])
            @test peel(U) ≈ (1 + Δt) .* phi atol = 1e-10
            xy = SVector(0.2, 0.3)
            @test evaldiskblock(U, Δt, Δt, xy; beta = beta) ≈
                  (1 + Δt) * evaldiskblock(reshape(phi, 1, :), 1.0, 0.0, xy; beta = beta) atol = 1e-10
        end
    end

    @testset "Zernike peel-off equals the final disk slice" begin
        Nt, nblocks, Δt, beta = 5, 5, 0.25, 0.5
        K = zernike_ncoeffs(nblocks)
        U = zeros(Nt, K)
        U[1, 1] = 0.7; U[2, 2] = -0.2; U[3, 4] = 0.1; U[4, 7] = -0.08; U[5, 11] = 0.05
        final_slice = reshape(peel(U), 1, :)
        for xy in (SVector(0.0, 0.0), SVector(0.2, 0.3), SVector(-0.4, 0.1), SVector(0.1, -0.5))
            @test evaldiskblock(final_slice, 1.0, 0.0, xy; beta = beta) ≈ evaldiskblock(U, Δt, Δt, xy; beta = beta)
        end
    end
end
