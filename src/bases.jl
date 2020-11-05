import Base: adjoint, getindex
import Plots: plot

abstract type Basis end

abstract type periodic end # What is the super type, BC?

struct BSplineBasis{T} <: Basis where {T}
    B #::BT need to define proper type, contains the basis functions
    t::AbstractVector # the knotvector
    interval::AbstractInterval # domain
    k::Int # order
    N::Int # also equal to length(B), do we need this?, amount of elements
end

getindex(B::BSplineBasis{T}, x::Real, j::Integer) where T = B.B[j](x)
getindex(B::BSplineBasis{T}, x::AbstractVector, j::Integer) where T = B.B[j].(x)

# Calculate the Spline Basis of order k
# For the Splines, we use CompactBases.jl. For the future, we should write our own libary with a better interface. 
function PeriodicBSpline(interval::AbstractInterval, k::Int, N::Int)
    a = interval.left
    b = interval.right
    Δx = (b - a)/N

    # Create a KnotSequence for the cardinal B-Splines
    t_ = range(a-k*Δx, b+k*Δx, step=Δx)
    t = ArbitraryKnotSet(k, t_)
    B = BSpline(t)

    basis_fct = []
    # Make the functions periodic 
    for i in k+1:2*k-1
        push!(basis_fct, x -> B[x, i] + B[x, N+i])
    end

    for i in 2*k:N+k
        push!(basis_fct, x -> B[x, i])
    end

    return BSplineBasis{periodic}(SVector(basis_fct...), t_, interval, k, N)
end

function derivative(basis::BSplineBasis{periodic})
    k = basis.k-1
    t_d = ArbitraryKnotSet(k, basis.t)
    B_d = BSpline(t_d)

    interval = basis.interval
    a = interval.left
    b = interval.right
    N = basis.N
    Δx = (b - a)/N

    basis_fct = []
    for i in k+2:2*k
        push!(basis_fct, x -> 1/Δx * (B_d[x, i] + B_d[x, N+i]))
    end

    for i in 2*k+1:N+k+1
        push!(basis_fct, x -> 1/Δx * B_d[x, i])
    end
    
    return BSplineBasis{periodic}(SVector(basis_fct...), basis.t, interval, k, N)
end

function adjoint(basis::Basis)
    return derivative(basis)
end

function plot(basis::Basis)
    a = basis.interval.left
    b = basis.interval.right
    xx = range(a, b, length=1000)
    fig = plot()
    for i in 1:length(basis.B)
        plot!(fig, xx, basis[xx, i])
    end
    
    return fig
end

# You can do this much more elegant, using flattening...
abstract type TensorProductBasis{N} end

# Just collect the 1-D bases as a tuple
struct BSplineTensorProductBasis{N, T} <: TensorProductBasis{N}
    B::SArray{Tuple{N}, BSplineBasis{T}}
end

getindex(B::BSplineTensorProductBasis{N, T}, j::Integer) where {N, T} = B.B[j]
getindex(B::BSplineTensorProductBasis{N, T}, j::Colon) where {N, T} = B.B

function PeriodicBSplineTensorProductBasis(domain::ProductDomain, k::NTuple{d,Int}, N::NTuple{d,Int}) where {d}  
    basis = []
    for i in 1:d
        push!(basis, PeriodicBSpline(domain.domains[i], k[i], N[i]))
    end
    BSplineTensorProductBasis{d, periodic}(SVector(basis...))
end

function tensorproduct(a::BSplineBasis{T}, b::BSplineBasis{T}) where T
    Bs = SVector(a, b)
    BSplineTensorProductBasis{2, T}(SVector(a, b))
end

function tensorproduct(a::BSplineTensorProductBasis{d, T}, b::BSplineBasis{T}) where {d, T}
    B = SVector(a.B..., b)
    BSplineTensorProductBasis{d+1, T}(B)
end

function tensorproduct(a::BSplineBasis{T}, b::BSplineTensorProductBasis{d, T}) where {d, T}
    B = SVector(a, b.B...)
    BSplineTensorProductBasis{d+1, T}(B)
end

function tensorproduct(a::BSplineTensorProductBasis{d1, T}, b::BSplineTensorProductBasis{d2, T}) where {d1, d2, T}
    B = SVector(a.B..., b.B...)
    BSplineTensorProductBasis{d1+d2, T}(B)
end

⊗(a::Union{TensorProductBasis, Basis}, b::Union{TensorProductBasis, Basis}) = tensorproduct(a, b)

function derivative(basis::BSplineTensorProductBasis{d, T}) where {d, T}
    b = derivative(basis.B[1])
    for i in 2:d
        bn = derivative(basis.B[i])
        b = b ⊗ bn
    end

    return b
end

function adjoint(basis::TensorProductBasis)
    return derivative(basis)
end