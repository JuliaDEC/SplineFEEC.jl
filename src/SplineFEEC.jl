module SplineFEEC

using CompactBases # Used for basis creation
using DomainSets
import DomainSets: (..), ×

using LinearAlgebra
using StaticArrays
using FastGaussQuadrature
using Kronecker
import Kronecker: ⊗
using FFTW

using Plots
using Cubature

export (..), ×, ⊗, ⊙, adjoint,
        PeriodicBSpline, DirichletBSpline, PeriodicBSplineTensorProductBasis, DirichletBSplineTensorProductBasis,
        mass_matrix, stiffness_matrix, L2_prod, col_mat, L2_proj, weighted_mass_matrix, interior_product_matrix,
        grad, curl, div, ∇, disc_diff, disc_grad, disc_curl, disc_div,
        Basis, TensorProductBasis, BSplineTensorProductBasis,
        DirichletMultipatchBSpline, mass_matrix_d, L2_prod_d, L2_proj_d, col_mat_d

include("bases.jl")
include("circulant.jl")
include("util.jl")
include("operator.jl")
include("multipatch.jl")
include("examples.jl")

end