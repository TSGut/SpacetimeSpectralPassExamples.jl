
[![DOI](https://zenodo.org/badge/1281489246.svg)](https://doi.org/10.5281/zenodo.20938998)

# SpacetimeSpectralPassExamples.jl

**Paper:** [TODO: add link]

This repository demonstrates the coefficient-level peel-and-pass for sparse space-time spectral methods with a Legendre basis in time. A space-time
block is solved on a bounded time interval, the final time slice is recovered by
an exact contraction over the time coefficients, and those spatial coefficients
are passed as initial data for the next block.

This is a companion collection of example scripts, one per numerical experiment
in the paper, not a registered or general-purpose package.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/TSGut/SpacetimeSpectralPassExamples.jl")
```

## Examples

Each experiment is one script under `examples/`, run with the package environment
active. It writes its figures and a CSV data cache under `examples/Figures/`.

## Related Packages
The examples were implemented using
[ClassicalOrthogonalPolynomials.jl](https://github.com/JuliaApproximation/ClassicalOrthogonalPolynomials.jl)
and
[MultivariateOrthogonalPolynomials.jl](https://github.com/JuliaApproximation/MultivariateOrthogonalPolynomials.jl) which would likely also form the starting point for a more general implementation in Julia.
