function benchmark_ms(f)
    f()                       # warm up before timing
    GC.gc()
    secs = parse(Float64, get(ENV, "FIGURE_BENCHMARK_SECONDS", "0.25"))
    samps = parse(Int, get(ENV, "FIGURE_BENCHMARK_SAMPLES", "8"))
    trial = @benchmark $f() seconds = secs samples = samps evals = 1
    median(trial).time / 1e6
end

function heat_operators(Nt::Integer, Nx::Integer)
    get!(HEAT_OPERATOR_CACHE, (Int(Nt), Int(Nx))) do
        Dt = _sparse_matrix(_D_C32(Nt)); St = _sparse_matrix(_S_C32(Nt))
        D2x = _sparse_matrix(_D2_C52(Nx)); Sx = _sparse_matrix(_S_C52(Nx))
        em1t = _sparse_row(_evalm1(Nt))
        em1x = _sparse_row(_evalm1(Nx)); ep1x = _sparse_row(_evalp1(Nx))
        Ix = _sparse_eye(Nx); It = _sparse_eye(Nt)
        keep = vec([jt + (jx - 1) * Nt for jt in 1:Nt, jx in 1:Nx if jt <= Nt - 1 && jx <= Nx - 2])
        (; Dt, St, D2x, Sx, em1t, em1x, ep1x, Ix, It, keep)
    end
end

function heat_problem_rect(Δt::Real, Nt::Integer, Nx::Integer)
    (; Dt, St, D2x, Sx, em1t, em1x, ep1x, Ix, It, keep) = heat_operators(Nt, Nx)
    Apde = kron(Sx, (2 / Δt) * Dt) - kron(4 * D2x, St)
    A = [kron(Ix, em1t);
         kron(em1x, It);
         kron(ep1x, It);
         Apde[keep, :]]
    keeprows = setdiff(1:size(A, 1), [Nx - 1, Nx])
    (; A, keeprows, factor = lu(A[keeprows, :]), Nt, Nx, keep, nnzA = nnz(A),
     b = zeros(Nx + 2Nt + length(keep)), U = zeros(Nt, Nx))
end

function heat_rhs_rect(prob, t0::Real, Δt::Real; forcing_time, space_profile = sinpi)
    Ft = timecoeffs_basis(U32, forcing_time, t0, Δt, prob.Nt)
    Fx = spacecoeffs_basis(U52, space_profile, prob.Nx)
    vec(Ft * transpose(Fx))[prob.keep]
end

function heatblock_rect!(U::AbstractMatrix, prob, u0c::AbstractVector, rhskeep::AbstractVector)
    b = prob.b
    fill!(b, 0)
    copyto!(b, 1, u0c, 1, prob.Nx)
    copyto!(b, prob.Nx + 2prob.Nt + 1, rhskeep, 1, length(rhskeep))
    copyto!(U, reshape(prob.factor \ b[prob.keeprows], prob.Nt, prob.Nx))
    U
end

heatblock_rect(prob, u0c::AbstractVector, rhskeep::AbstractVector) =
    heatblock_rect!(similar(prob.U), prob, u0c, rhskeep)

heat_rhs_blocks(prob, Δt::Real, L::Integer; forcing_time, space_profile = sinpi) =
    [heat_rhs_rect(prob, (ell - 1) * Δt, Δt; forcing_time = forcing_time, space_profile = space_profile) for ell in 1:L]

function stepsolve_heat_rect(u0c::AbstractVector, Δt::Real, L::Integer, Nt::Integer, Nx::Integer; rhsblocks)
    prob = heat_problem_rect(Δt, Nt, Nx)
    c = collect(float.(u0c))
    for ell in 1:L
        heatblock_rect!(prob.U, prob, c, rhsblocks[ell])
        peel!(c, prob.U)
    end
    c
end

function wave_operators(Nt::Integer, Nx::Integer)
    get!(WAVE_OPERATOR_CACHE, (Int(Nt), Int(Nx))) do
        Dt = _sparse_matrix(_D_C32(Nt)); St = _sparse_matrix(_S_C32(Nt))
        D2x = _sparse_matrix(_D2_C52(Nx)); Sx = _sparse_matrix(_S_C52(Nx))
        em1t = _sparse_row(_evalm1(Nt))
        em1x = _sparse_row(_evalm1(Nx)); ep1x = _sparse_row(_evalp1(Nx))
        Ix = _sparse_eye(Nx); It = _sparse_eye(Nt)
        keep_all = vec([jt + (jx - 1) * Nt for jt in 1:Nt, jx in 1:Nx if jt <= Nt - 1])
        keep_interior = vec([jt + (jx - 1) * Nt for jt in 1:Nt, jx in 1:Nx if jt <= Nt - 1 && jx <= Nx - 2])
        (; Dt, St, D2x, Sx, em1t, em1x, ep1x, Ix, It, keep_all, keep_interior)
    end
end

function wave_problem_rect(Δt::Real, Nt::Integer, Nx::Integer; m2::Real = 0.0)
    (; Dt, St, D2x, Sx, em1t, em1x, ep1x, Ix, It, keep_all, keep_interior) = wave_operators(Nt, Nx)
    n = Nt * Nx
    init_u = [kron(Ix, em1t) spzeros(Nx, n)]
    init_v = [spzeros(Nx, n) kron(Ix, em1t)]
    bdy = [[kron(em1x, It); kron(ep1x, It)] spzeros(2Nt, n)]
    eq1_u = kron(Ix, (2 / Δt) * Dt); eq1_v = -kron(Ix, St)
    eq2_u = -4 * kron(D2x, St) + m2 * kron(Sx, St); eq2_v = kron(Sx, (2 / Δt) * Dt)
    A = [init_u; init_v; bdy;
         [eq1_u[keep_all, :] eq1_v[keep_all, :]];
         [eq2_u[keep_interior, :] eq2_v[keep_interior, :]]]
    keeprows = setdiff(1:size(A, 1), [Nx - 1, Nx])
    (; A, keeprows, factor = lu(A[keeprows, :]), Nt, Nx, Δt, keep_all, keep_interior, nnzA = nnz(A),
     b = zeros(size(A, 1)), U = zeros(Nt, Nx), V = zeros(Nt, Nx))
end

function waveblock_rect!(U::AbstractMatrix, V::AbstractMatrix, prob, u0c::AbstractVector, v0c::AbstractVector)
    n = prob.Nt * prob.Nx
    b = prob.b
    fill!(b, 0)
    copyto!(b, 1, u0c, 1, prob.Nx)
    copyto!(b, prob.Nx + 1, v0c, 1, prob.Nx)
    sol = prob.factor \ b[prob.keeprows]
    copyto!(U, reshape(view(sol, 1:n), prob.Nt, prob.Nx))
    copyto!(V, reshape(view(sol, n + 1:2n), prob.Nt, prob.Nx))
    U, V
end

function stepsolve_wave_rect(u0c::AbstractVector, v0c::AbstractVector, Δt::Real, L::Integer, Nt::Integer, Nx::Integer; m2::Real = 0.0)
    prob = wave_problem_rect(Δt, Nt, Nx; m2 = m2)
    u = collect(float.(u0c)); v = collect(float.(v0c))
    for _ in 1:L
        waveblock_rect!(prob.U, prob.V, prob, u, v)
        peel!(u, prob.U)
        peel!(v, prob.V)
    end
    u, v
end

function wave_scalar_operators(Nt::Integer, Nx::Integer)
    get!(WAVE_SCALAR_OPERATOR_CACHE, (Int(Nt), Int(Nx))) do
        D2t = _sparse_matrix(_D2_C52(Nt)); St = _sparse_matrix(_S_C52(Nt))
        D2x = _sparse_matrix(_D2_C52(Nx)); Sx = _sparse_matrix(_S_C52(Nx))
        em1t = _sparse_row(_evalm1(Nt)); dem1t = _sparse_row(_devalm1(Nt))
        em1x = _sparse_row(_evalm1(Nx)); ep1x = _sparse_row(_evalp1(Nx))
        Ix = _sparse_eye(Nx); It = _sparse_eye(Nt)
        keep = vec([jt + (jx - 1) * Nt for jt in 1:Nt, jx in 1:Nx if jt <= Nt - 2 && jx <= Nx - 2])
        (; D2t, St, D2x, Sx, em1t, dem1t, em1x, ep1x, Ix, It, keep)
    end
end

function wave_scalar_problem_rect(Δt::Real, Nt::Integer, Nx::Integer; m2::Real = 0.0)
    (; D2t, St, D2x, Sx, em1t, dem1t, em1x, ep1x, Ix, It, keep) = wave_scalar_operators(Nt, Nx)
    Apde = (4 / Δt^2) * kron(Sx, D2t) - 4 * kron(D2x, St) + m2 * kron(Sx, St)
    A = [kron(Ix, em1t);
         kron(Ix, (2 / Δt) * dem1t);
         kron(em1x, It);
         kron(ep1x, It);
         Apde[keep, :]]
    rownorms = vec(sqrt.(sum(abs2, A; dims = 2)))
    d = [r > 0 ? inv(r) : 1.0 for r in rownorms]
    A = Diagonal(d) * A
    keeprows = setdiff(1:size(A, 1), [Nx - 1, Nx, 2Nx - 1, 2Nx])
    (; factor = lu(A[keeprows, :]), keeprows, Nt, Nx, Δt, keep, nnzA = nnz(A), d,
     b = zeros(2Nx + 2Nt + length(keep)), U = zeros(Nt, Nx))
end

function wave_scalar_block_rect!(U::AbstractMatrix, prob, u0c::AbstractVector, v0c::AbstractVector)
    b = prob.b
    fill!(b, 0)
    copyto!(b, 1, u0c, 1, prob.Nx)
    copyto!(b, prob.Nx + 1, v0c, 1, prob.Nx)
    b .*= prob.d   # match the row scaling
    copyto!(U, reshape(prob.factor \ b[prob.keeprows], prob.Nt, prob.Nx))
    U
end

function stepsolve_wave_scalar_rect(u0c::AbstractVector, v0c::AbstractVector, Δt::Real, L::Integer, Nt::Integer, Nx::Integer; m2::Real = 0.0)
    prob = wave_scalar_problem_rect(Δt, Nt, Nx; m2 = m2)
    u = collect(float.(u0c)); v = collect(float.(v0c))
    for _ in 1:L
        wave_scalar_block_rect!(prob.U, prob, u, v)
        peel!(u, prob.U)
        peel_dt!(v, prob.U, Δt)
    end
    u, v
end

function disk_operator_matrix(Nt, nblocks; beta = 1.0)
    K = zernike_ncoeffs(nblocks)
    C, L = disk_operators_cached(nblocks; beta = beta)
    Dt, St = Matrix(_D_C32(Nt)), Matrix(_S_C32(Nt))
    em1 = _evalm1(Nt)
    Apde = kron(C, 2 * Dt) + kron(L, St)
    keep = vec([jt + (jx - 1) * Nt for jt in 1:Nt, jx in 1:K if jt <= Nt - 1])
    sparse_drop([kron(I(K), transpose(em1)); Apde[keep, :]])
end

function disk_problem_sparse(beta::Real, N::Integer, Δt::Real, nblocks::Integer)
    get!(DISK_PROBLEM_CACHE, (Float64(beta), Int(N), Float64(Δt))) do
        K = zernike_ncoeffs(nblocks)
        C, L = disk_operators_cached(nblocks; beta = beta)
        Dt, St = Matrix(_D_C32(N)), Matrix(_S_C32(N))
        em1 = _evalm1(N)
        Apde = kron(C, (2 / Δt) * Dt) + kron(L, St)
        keep = vec([jt + (jx - 1) * N for jt in 1:N, jx in 1:K if jt <= N - 1])
        A = sparse_drop([kron(I(K), transpose(em1)); Apde[keep, :]])
        (; A, keep, Nt = N, K, Δt, nnzA = stored_nnz(A),
         b = zeros(K + length(keep)), U = zeros(N, K))
    end
end

function disk_apply_factor!(U::AbstractMatrix, prob, factor, u0c::AbstractVector, rhskeep::AbstractVector)
    fill!(prob.b, 0)
    copyto!(prob.b, 1, u0c, 1, prob.K)
    copyto!(prob.b, prob.K + 1, rhskeep, 1, length(rhskeep))
    copyto!(U, reshape(factor \ prob.b, prob.Nt, prob.K))
    U
end

function disk_solve_from_sparse(prob, u0c::AbstractVector, rhsblocks)
    factor = lu(prob.A)
    c = collect(float.(u0c))
    U = similar(prob.U)
    for rhskeep in rhsblocks
        disk_apply_factor!(U, prob, factor, c, rhskeep)
        peel!(c, U)
    end
    U
end

float_storage_count(x::AbstractArray{<:AbstractFloat}) = length(x)
float_storage_count(x::SparseMatrixCSC{<:AbstractFloat}) = nnz(x)
float_storage_count(x::Number) = x isa AbstractFloat ? 1 : 0
function float_storage_count(x)
    isstructtype(typeof(x)) || return 0
    sum(float_storage_count(getfield(x, n)) for n in fieldnames(typeof(x)); init = 0)
end

_factor_floats(F::SparseArrays.UMFPACK.UmfpackLU) = nnz(F.L) + nnz(F.U)
_factor_floats(F) = nnz(F.R)

function resident_floats(prob, state_floats::Integer)
    factor_floats = hasproperty(prob, :factor) ? _factor_floats(prob.factor) : 0
    factor_floats + length(prob.b) + length(prob.U) + state_floats
end

heat_memory(Nt, Nx, Δt) = resident_floats(heat_problem_rect(Δt, Nt, Nx), Nx)
function wave_memory(Nt, Nx, Δt; m2 = 0.0)
    prob = wave_problem_rect(Δt, Nt, Nx; m2 = m2)
    resident_floats(prob, 2Nx) + length(prob.V)
end
wave_scalar_memory(Nt, Nx, Δt; m2 = 0.0) = resident_floats(wave_scalar_problem_rect(Δt, Nt, Nx; m2 = m2), 2Nx)

disk_memory(prob) = _factor_floats(lu(prob.A)) + length(prob.b) + length(prob.U) + prob.K

function heat_experiment_data(Nts_by_L; Nx, T, time_profile, forcing_time, exact, space_profile = sinpi)
    rows = Float64[]
    c0 = time_profile(0.0) .* spacecoeffs(space_profile, Nx)
    for (L, Nts) in Nts_by_L
        dt = T / L
        for Nt in Nts
            prob = heat_problem_rect(dt, Nt, Nx)
            rhsblocks = heat_rhs_blocks(prob, dt, L; forcing_time = forcing_time, space_profile = space_profile)
            t_ms = benchmark_ms(() -> stepsolve_heat_rect(c0, dt, L, Nt, Nx; rhsblocks = rhsblocks))
            cT = stepsolve_heat_rect(c0, dt, L, Nt, Nx; rhsblocks = rhsblocks)
            err = maximum(abs(evalspace(cT, x) - exact(T, x)) for x in GRIDX)
            append!(rows, (L, Nt, t_ms, err, heat_memory(Nt, Nx, dt)))
        end
    end
    reshape(rows, 5, :)'
end

function heat_local_error_data(Nts_by_L; Nx, T, time_profile, forcing_time, exact, space_profile = sinpi)
    rows = Float64[]
    sx = spacecoeffs(space_profile, Nx)
    for (L, Nts) in Nts_by_L
        dt = T / L
        for Nt in Nts
            prob = heat_problem_rect(dt, Nt, Nx)
            worst = 0.0
            for ell in 1:L
                t0 = (ell - 1) * dt
                rhs = heat_rhs_rect(prob, t0, dt; forcing_time = forcing_time, space_profile = space_profile)
                cT = peel(heatblock_rect(prob, time_profile(t0) .* sx, rhs))
                worst = max(worst, maximum(abs(evalspace(cT, x) - exact(ell * dt, x)) for x in GRIDX))
            end
            append!(rows, (L, Nt, worst))
        end
    end
    reshape(rows, 3, :)'
end

function wave_experiment_data(Nts_by_L; m2 = 0.0, exact, Nx, T, init_profile = x -> sinpi(2x))
    rows = Float64[]
    c0 = collect(spacecoeffs(init_profile, Nx)); v0 = zeros(Nx)
    for (L, Nts) in Nts_by_L
        dt = T / L
        for Nt in Nts
            t_ms = benchmark_ms(() -> stepsolve_wave_rect(c0, v0, dt, L, Nt, Nx; m2 = m2))
            uT, _ = stepsolve_wave_rect(c0, v0, dt, L, Nt, Nx; m2 = m2)
            err = maximum(abs(evalspace(uT, x) - exact(T, x)) for x in GRIDX)
            append!(rows, (L, Nt, t_ms, err, wave_memory(Nt, Nx, dt; m2 = m2)))
        end
    end
    reshape(rows, 5, :)'
end

function wave_scalar_experiment_data(Nts_by_L; m2 = 0.0, exact, Nx, T, init_profile = x -> sinpi(2x))
    rows = Float64[]
    c0 = collect(spacecoeffs(init_profile, Nx)); v0 = zeros(Nx)
    for (L, Nts) in Nts_by_L
        dt = T / L
        for Nt in Nts
            t_ms = benchmark_ms(() -> stepsolve_wave_scalar_rect(c0, v0, dt, L, Nt, Nx; m2 = m2))
            uT, _ = stepsolve_wave_scalar_rect(c0, v0, dt, L, Nt, Nx; m2 = m2)
            err = maximum(abs(evalspace(uT, x) - exact(T, x)) for x in GRIDX)
            append!(rows, (L, Nt, t_ms, err, wave_scalar_memory(Nt, Nx, dt; m2 = m2)))
        end
    end
    reshape(rows, 5, :)'
end
