import Base: adjoint, getindex
import Plots: plot

abstract type Basis end

abstract type periodic end # What is the super type, BC?

"""
    BSplineBasis{T}(basis_functions, knot_vector, interval, order, N)

    BSpline basis on `interval` with `N` points and order `k`. Where `T` describes the boundary condition.
"""
struct BSplineBasis{T} <: Basis where {T}
    B #::BT need to define proper type, contains the basis functions
    t::AbstractVector # the knotvector
    interval::AbstractInterval # domain
    k::Int # order
    N::Int # also equal to length(B), do we need this?, amount of elements
end

function Base.show(io::IO, B::BSplineBasis{T}) where {T}
    println(io, T, " BSpline basis of order ", B.k, " on interval ", B.interval, " with ", B.N, " nodes.")
    nothing
end

getindex(B::BSplineBasis{T}, x::Real, j::Integer) where T = B.B[j](x)
getindex(B::BSplineBasis{T}, x::Real, j::Colon) where T = [B.B[i](x) for i in 1:length(B.B)]
getindex(B::BSplineBasis{T}, x::AbstractVector, j::Integer) where T = B.B[j].(x)
getindex(B::BSplineBasis{T}, x::AbstractVector, j::Colon) where T = [B.B[i](x[j]) for j in 1:length(x), i in 1:length(B.B)]

# Calculate the Spline Basis of order k
# For the Splines, we use CompactBases.jl. For the future, we should write our own libary with a better interface. 
"""
    PeriodicBSpline(basis_functions, knot_vector, interval, order, N)

    Construct a `periodic` BSpline basis on `interval` with `N` points and order `k`.
"""
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

    # In order to have the same ordering as in theory
    basis_fct = circshift(basis_fct, -1)

    return BSplineBasis{periodic}(SVector(basis_fct...), t_, interval, k, N)
end

# For the second derivative, the last basis functions has problems with the BC
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

"""
    adjoint(b::Basis)

    Create a basis that consits of the derivative of basis functions in `b`.
"""
function adjoint(basis::Basis)
    return derivative(basis)
end

# we can delete this later on
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
"""
    BSplineTensorProductBasis{N,T}(Bases::SArray)

    BSpline basis on a tensor-product domain, consisting of a vector of one-dimensional bases defined on each direction of the tensor-product domain.
    Where `N` describes the dimension and `T` the boundary condition.
"""
struct BSplineTensorProductBasis{N, T} <: TensorProductBasis{N}
    B::SArray{Tuple{N}, BSplineBasis{T}}
end

function Base.show(io::IO, B::BSplineTensorProductBasis{N,T}) where {N,T}
    domain = B[1].interval
    for i in 2:N
        domain = domain × B[i].interval
    end

    println(io, T, " BSpline basis on tensor-product domain ", domain, ", consisting of: ")
    for i in 1:N
        show(io::IO, B.B[i])
    end
    nothing
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