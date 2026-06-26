const P   = Legendre()
const U32 = Ultraspherical(3/2)   # derivative target  P -> C^{3/2}
const U52 = Ultraspherical(5/2)   # 2nd-derivative target P -> C^{5/2}

_D_C32(N)  = (U32 \ (Derivative(axes(P,1))   * P))[1:N, 1:N]  # P -> C^{3/2}, d/dξ
_S_C32(N)  = (U32 \ P)[1:N, 1:N]                              # P -> C^{3/2} conversion
_D2_C52(N) = (U52 \ (Derivative(axes(P,1))^2 * P))[1:N, 1:N]  # P -> C^{5/2}, d²/dξ²
_S_C52(N)  = (U52 \ P)[1:N, 1:N]                              # P -> C^{5/2} conversion
_evalm1(N) = P[-1.0, 1:N]                                     # P_j(-1) = (-1)^j
_evalp1(N) = P[ 1.0, 1:N]                                     # P_j(1)  = 1
_devalm1(N) = (Derivative(axes(P,1)) * P)[-1.0, 1:N]          # P_j'(-1)

_sparse_matrix(A) = sparse(Matrix(A))
_sparse_row(v) = sparse(reshape(collect(v), 1, length(v)))
_sparse_eye(N) = sparse(I, N, N)

stored_nnz(A) = A isa SparseMatrixCSC ? nnz(A) : count(!iszero, A)

# Drop entries of magnitude at most `tol`, returning a pruned sparse matrix.
function sparse_drop(A; tol = 1e-15)
    B = Matrix(A)
    B[abs.(B) .<= tol] .= 0
    dropzeros!(sparse(B))
end

const HEAT_OPERATOR_CACHE = Dict{Tuple{Int,Int},Any}()
const WAVE_OPERATOR_CACHE = Dict{Tuple{Int,Int},Any}()
const WAVE_SCALAR_OPERATOR_CACHE = Dict{Tuple{Int,Int},Any}()
const DISK_OPERATOR_CACHE = Dict{Tuple{Int,Float64},Any}()
const DISK_PROBLEM_CACHE = Dict{Tuple{Float64,Int,Float64},Any}()

const GRIDX = range(0, 1, 201)

zernike_ncoeffs(nblocks::Integer) = div(nblocks * (nblocks + 1), 2)

function _disk_sample_points(K::Integer)
    nrad = max(4, ceil(Int, sqrt(K)) + 3)
    nang = max(16, 2nrad + 4)
    pts = SVector{2,Float64}[]
    for i in 1:nrad
        r = 0.98 * (i - 0.5) / nrad
        for j in 1:nang
            θ = 2π * (j - 1) / nang
            push!(pts, SVector(r * cos(θ), r * sin(θ)))
        end
    end
    pts
end

function _weighted_zernike_conversion(Z, WZ, K::Integer; beta::Real)
    if isinteger(beta)
        return Matrix((Z \ WZ)[1:K, 1:K])
    end

    pts = _disk_sample_points(K)
    VZ = Matrix{Float64}(undef, length(pts), K)
    VWZ = similar(VZ)
    for (i, xy) in pairs(pts)
        VZ[i, :] .= Z[xy, 1:K]
        VWZ[i, :] .= WZ[xy, 1:K]
    end
    VZ \ VWZ
end

function _disk_operators(nblocks::Integer; beta::Real = 1.0)
    beta > 0 || throw(ArgumentError("disk heat blocks require beta > 0"))
    K = zernike_ncoeffs(nblocks)
    Z = Zernike(beta)
    WZ = Weighted(Z)
    C = _weighted_zernike_conversion(Z, WZ, K; beta = beta)
    # diagonal on weighted Zernike; slice finite first, keep Diagonal
    L = Diagonal(diag((Z \ (AbsLaplacian(WZ, beta) * WZ))[1:K, 1:K]))
    C, L
end

disk_operators_cached(nblocks::Integer; beta::Real = 1.0) =
    get!(DISK_OPERATOR_CACHE, (Int(nblocks), Float64(beta))) do
        _disk_operators(nblocks; beta = beta)
    end

function peel!(out::AbstractVector, U::AbstractMatrix)
    length(out) == size(U, 2) || throw(DimensionMismatch("out has length $(length(out)), expected $(size(U, 2))"))
    fill!(out, zero(eltype(out)))
    # sum the small high-degree tail first
    @inbounds for k in axes(U, 2), j in reverse(axes(U, 1))
        out[k] += U[j, k]
    end
    out
end

peel(U::AbstractMatrix) = peel!(similar(U, size(U, 2)), U)

function peel_dt!(out::AbstractVector, U::AbstractMatrix, Δt::Real)
    length(out) == size(U, 2) || throw(DimensionMismatch("out has length $(length(out)), expected $(size(U, 2))"))
    fill!(out, zero(eltype(out)))
    # sum the small high-degree tail first
    @inbounds for k in axes(U, 2), i in reverse(axes(U, 1))
        j = i - 1
        out[k] += (j * (j + 1) / Δt) * U[i, k]
    end
    out
end

peel_dt(U::AbstractMatrix, Δt::Real) =
    peel_dt!(Vector{typeof(zero(eltype(U)) / Δt)}(undef, size(U, 2)), U, Δt)

spacecoeffs(f, N) = coefficients(expand(P, ξ -> f((ξ + 1) / 2)))[1:N]

timecoeffs_basis(B, f, t0, dt, N) = coefficients(expand(B, τ -> f(t0 + dt * (τ + 1) / 2)))[1:N]
timecoeffs_legendre(f, t0, dt, N) = timecoeffs_basis(Legendre(), f, t0, dt, N)
spacecoeffs_basis(B, f, N) = coefficients(expand(B, ξ -> f((ξ + 1) / 2)))[1:N]

evalspace(c::AbstractVector, x::Real) = P[2x - 1, 1:length(c)]' * c

function evalblock(U::AbstractMatrix, Δt::Real, t::Real, x::Real)
    Nt, Nx = size(U)
    P[2t / Δt - 1, 1:Nt]' * U * P[2x - 1, 1:Nx]
end

function evalblock_final(U::AbstractMatrix, x::Real)
    Nt, Nx = size(U)
    P[1.0, 1:Nt]' * U * P[2x - 1, 1:Nx]
end

function evaldiskblock(U::AbstractMatrix, Δt::Real, t::Real, xy; beta::Real = 1.0)
    Nt, K = size(U)
    dot(P[2t / Δt - 1, 1:Nt], U * Weighted(Zernike(beta))[xy, 1:K])
end
