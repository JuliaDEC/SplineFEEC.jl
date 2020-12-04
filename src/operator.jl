### L2
function L2_prod(basis::Union{Basis, TensorProductBasis})
    return mass_matrix(basis)
end

function L2_prod(basis1::Basis, basis2::Basis)
    return mass_matrix(basis1, basis2)
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
        L2_prod(a, b)
    end
end

### derivative bases
# Maybe need to generalize this to d1, d2, d3,...
abstract type grad end
abstract type curl end
abstract type div end

struct d_basis{D, T <: Union{Basis, TensorProductBasis}}
    basis::T
end

function grad(basis::Union{Basis, TensorProductBasis})
    return d_basis{grad, typeof(basis)}(basis)
end
function ∇(basis::Union{Basis, TensorProductBasis})
    return d_basis{grad, typeof(basis)}(basis)
end

function L2_prod(gbasis::d_basis{grad, T}) where T <: Basis
    return stiffness_matrix(derivative(gbasis.basis))
end

function L2_prod(gbasis::d_basis{grad, T}) where T <: TensorProductBasis{N} where N
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

function ⊙(a::Union{d_basis{grad, T}, d_basis{grad, D}}, b::Union{d_basis{grad, T}, d_basis{grad, D}}) where {T <: Basis, D <: TensorProductBasis}
    if a == b
        return L2_prod(a)
    else
        error("todo")
    end
end

### discrete differentials
function der_mat(N::Int)
    c = zeros(N)
    c[1] = 1
    c[2] = -1

    return Circulant(c)
end

function BSplineDeRham_DiffMatrices(N::SArray{Tuple{3}})
    D1 = der_mat(N[1]) ⊗ circ_id(N[2]) ⊗ circ_id(N[3])
    D2 = circ_id(N[1]) ⊗ der_mat(N[2]) ⊗ circ_id(N[3])
    D3 = circ_id(N[1]) ⊗ circ_id(N[2]) ⊗ der_mat(N[3])

    G = [D1; D2; D3]
    C = [zero(D1) -D3 D2;
        D3  zero(D2) -D1;
        -D2 D1 zero(D3)]
    D = [D1 D2 D3]

    return G, C, D
end

#This way its hard to generalize the notation I think..
#Would be better to give them names like d{t, N} corresponding to d^t in N dimensions
struct discrete_differential{T, N} 
    D::SArray{Tuple{N}} #1d diff matrices
end

function disc_grad(N)
    return discrete_diff(gradient, N)
end

function disc_curl(N)
    return discrete_diff(curl, N)
end

function disc_div(N)
    return discrete_diff(div, N)
end

#2D case
#here we only have gradient and curl in the primal diagram, curl and div in the dual diagram, which we write similarly
#todo add dual operators properly
function disc_diff(T, N::NTuple{2})
    D1 = der_mat(N[1]) ⊗ circ_id(N[2])
    D2 = circ_id(N[1]) ⊗ der_mat(N[2])

    return discrete_differential{T, 2}(SVector(D1, D2))
end

function *(C::discrete_differential{gradient, 2}, fh::SArray)
    return SVector(C.D[1] * fh, C.D[2] * fh)
end

function *(C::discrete_differential{curl, 2}, fh::SArray)
    return SVector{length(fh[1])}(C.D[1] * fh[2] - C.D[2] * fh[1])
end

function matrix(C::discrete_differential{gradient, 2})
    return [C.D[1];
            C.D[2]]
end

function matrix(C::discrete_differential{curl, 2})
    return [-1 .* C.D[2] C.D[1]]
end

function adjoint(C::discrete_differential{gradient, 2})
    return discrete_differential{curl, 2}(SVector(C.D[2]',  -1 .* C.D[1]'))
end

function adjoint(C::discrete_differential{curl, 2})
    return discrete_differential{gradient, 2}(SVector( -1 .* C.D[2]', C.D[1]'))
end

#3D case
# brackets so that the kronecker product ordering coincides with the mass matrix
function disc_diff(T, N::NTuple{3})
    D1 = der_mat(N[1]) ⊗ (circ_id(N[2]) ⊗ circ_id(N[3]))
    D2 = circ_id(N[1]) ⊗ (der_mat(N[2]) ⊗ circ_id(N[3]))
    D3 = circ_id(N[1]) ⊗ (circ_id(N[2]) ⊗ der_mat(N[3]))

    return discrete_differential{T, 3}(SVector(D1, D2, D3))
end

function *(C::discrete_differential{gradient, 3}, fh::SArray)
    return SVector(C.D[1] * fh, C.D[2] * fh , C.D[3] * fh)
end

function *(C::discrete_differential{curl, 3}, fh::SArray{Tuple{3}})
    return SVector(C.D[3] * fh[2] - C.D[2] * fh[3], -C.D[3]* fh[1] + C.D[1] * fh[3], C.D[2] * fh[1] - C.D[1] * fh[2])
end

function *(C::discrete_differential{div, 3}, fh::SArray{Tuple{3}})
    return SVector(C.D[1] * fh[1] + C.D[2] * fh[2] + C.D[3] * fh[3])
end

function matrix(C::discrete_differential{gradient, 3})
    return [C.D[1];
            C.D[2];
            C.D[3]]
end

function matrix(C::discrete_differential{curl, 3})
    return [zero(C.D[1]) -C.D[3] C.D[2];
            C.D[3]  zero(C.D[2]) -C.D[1];
            -C.D[2] C.D[1] zero(C.D[3])]
end

function matrix(C::discrete_differential{div, 3})
    return [C.D[1] C.D[2] C.D[3]]
end

#since we don't have a matrix, we have to manually implement the transpose
function adjoint(C::discrete_differential{gradient, 3})
    return discrete_differential{div, 3}(SVector(C.D[1]', C.D[2]', C.D[3]'))
end

function adjoint(C::discrete_differential{curl, 3})
    return discrete_differential{curl, 3}(-1 .* SVector(C.D[1]', C.D[2]', C.D[3]'))
end

function adjoint(C::discrete_differential{div, 3})
    return discrete_differential{grad, 3}(SVector(C.D[1]', C.D[2]', C.D[3]'))
end