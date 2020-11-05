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
export (..), ×, ⊗

include("bases.jl")
include("circulant.jl")
include("util.jl")
include("operator.jl")
include("examples.jl")

end
