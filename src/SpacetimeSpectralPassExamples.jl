module SpacetimeSpectralPassExamples

using BenchmarkTools
using ClassicalOrthogonalPolynomials
import ClassicalOrthogonalPolynomials: expand, coefficients
using LinearAlgebra
using MultivariateOrthogonalPolynomials
using SparseArrays
using StaticArrays

include("core.jl")
include("blocks.jl")

# core.jl
export peel, peel!, peel_dt, peel_dt!,
       spacecoeffs, spacecoeffs_basis, timecoeffs_basis, timecoeffs_legendre,
       evalspace, evalblock, evalblock_final, evaldiskblock,
       zernike_ncoeffs, disk_operators_cached, GRIDX, U32, U52

# blocks.jl
export benchmark_ms,
       heat_problem_rect, heat_rhs_rect, heat_rhs_blocks, heatblock_rect, heatblock_rect!,
       stepsolve_heat_rect, wave_problem_rect, waveblock_rect!, stepsolve_wave_rect,
       wave_scalar_problem_rect, stepsolve_wave_scalar_rect,
       disk_operator_matrix, disk_problem_sparse, disk_apply_factor!, disk_solve_from_sparse,
       heat_memory, wave_memory, wave_scalar_memory, disk_memory, resident_floats, float_storage_count,
       heat_experiment_data, heat_local_error_data, wave_experiment_data, wave_scalar_experiment_data

end # module
