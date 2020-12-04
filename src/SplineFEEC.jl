module SplineFEEC

using CompactBases
using DomainSets
import DomainSets: (..), ×

using LinearAlgebra
using StaticArrays
using FastGaussQuadrature
using Kronecker
import Kronecker: ⊗
using FFTW

using Plots

# For now only export the non internal operators
export (..), ×, ⊗, ⊙, adjoint,
        PeriodicBSpline, PeriodicBSplineTensorProductBasis,
        mass_matrix, stiffness_matrix, L2_prod, col_mat, L2_proj,
        grad, curl, div, ∇, disc_diff, disc_grad, disc_curl, disc_div

include("bases.jl")
include("circulant.jl")
include("util.jl")
include("operator.jl")
include("examples.jl")

end