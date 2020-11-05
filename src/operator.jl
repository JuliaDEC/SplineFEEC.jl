function L2_prod(basis::Union{Basis, TensorProductBasis})
    return mass_matrix(basis)
end

function ⊙(f::Function, b::Union{Basis, TensorProductBasis}) 
    return L2_prod(f, b)
end

function ⊙(b::Union{Basis, TensorProductBasis}, f::Function)
    return L2_prod(f, b)
end

function ⊙(a::Union{Basis, TensorProductBasis}, b::Union{Basis, TensorProductBasis})
    if a == b
        return L2_prod(a)
    else
        error("todo") #need to define mass matrix with different basis functions
    end
end

function der_mat(N::Int)
    c = zeros(N)
    c[1] = -1
    c[end] = 1

    return Circulant(c)
end

struct grad{T <: Union{Basis, TensorProductBasis}}
    basis::T
end

function ∇(basis::Union{Basis, TensorProductBasis})
    return grad(basis)
end

function L2_prod(gbasis::grad{T}) where T <: Basis
    return stiffness_matrix(derivative(gbasis.basis))
end

function L2_prod(gbasis::grad{T}) where T <: TensorProductBasis{N} where N
    components = []
    basis = gbasis.basis
    for j in 1:N
        matrices = []
        for i in 1:N
            if i == j
                b = derivative(basis[i])
                push!(matrices, stiffness_matrix(b))
            else
                push!(matrices, mass_matrix(basis[i]))
            end
        end

        comp = ⊗(matrices...)
        push!(components, comp)
    end

    return sum(components)
end

function ⊙(a::Union{grad{T}, grad{D}}, b::Union{grad{T}, grad{D}}) where {T <: Basis, D <: TensorProductBasis}
    if a == b
        return L2_prod(a)
    else
        error("todo")
    end
end

struct curl{T <: TensorProductBasis{3}}
    basis::T
end

# How do we write ∇×, or define ∇ as a struct ?
function ∇x(basis::TensorProductBasis{3})
    return curl(basis)
end

struct div{T <: TensorProductBasis{3}}
    basis::T
end

# How do we write ∇⋅
function ∇o(basis::TensorProductBasis{3})
    return div(basis)
end