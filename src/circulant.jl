import Base: size, getindex, *, +, -, ^, \, adjoint, ones
import LinearAlgebra: inv, eigen

# Assuming we work with real matrices...
struct Circulant{T} <: AbstractArray{T, 2}
    co::Vector{T}
    fft::Vector{Complex{T}}
end

Circulant(co) = Circulant(co, fft(co))

size(c::Circulant) = length(c.co), length(c.co)
getindex(c::Circulant, i::Int, j::Int) = c.co[mod(i - j, length(c.co))+1]   

function *(a::Circulant, b::Circulant)
    return Circulant(real(ifft!(a.fft  .* b.fft)), a.fft  .* b.fft)
end

function *(a::Circulant, x::Vector)
     return ifft!(a.fft .* fft(x))
end

function *(a::Circulant, x::Vector{T}) where T <: Real
     return real(ifft!(a.fft .* fft(x)))
end

function +(a::Circulant, b::Circulant)
    return Circulant(a.co + b.co, a.fft + b.fft)
end

function -(a::Circulant, b::Circulant)
    return Circulant(a.co - b.co, a.fft - b.fft)
end

function ^(a::Circulant, n::Int)
    return Circulant(real(ifft!(a.fft.^n)), a.fft.^n)
end

function \(a::Circulant, x::Vector{T}) where T <: Real
     return real(ifft!(fft(x) ./ a.fft))
end

function inv(a::Circulant)
     return Circulant( real(ifft!(1 ./ a.fft )), 1 ./ a.fft)
end

function adjoint(a::Circulant)
    c = a.co
    c_new = vcat(c[1], c[end:-1:2])
    return Circulant(c_new)
end

function eigen(a::Circulant)
    λ = a.fft
    n = length(a.co)
    ω = [exp(2*im*π*k*j/n) for j in 0:n-1, k in 1:n]
    return λ, ω
end

function circ_ones(N::Int)
    return Circulant(ones(N))
end

function circ_id(N::Int)
    return Circulant([1, zeros(N-1)...])
end