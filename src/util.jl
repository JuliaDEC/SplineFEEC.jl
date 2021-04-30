# 1-D Calculations
"""
    mass_matrix(b::Basis)

    Calculate the mass-matrix for the basis `b` with Gauss quadrature. 
"""
function mass_matrix(basis::BSplineBasis{dirichlet})
    x, w = gausslegendre(basis.k + 10)
    a = basis.interval.left
    b = basis.interval.right
    N = length(basis.B)
    Δx = (b-a)/basis.N

    M = zeros(N, N)
    # Do Quadrature over every element -> exact
    for el in 1:basis.N
        b = a + Δx
        xx = a .+ (b - a) .*(x .+ 1) ./ 2

        # Otherwise sometimes the last element is wrong
        if xx[end] > basis.interval.right
            xx[end] = basis.interval.right
        end

        for i in 1:N
            for j in 1:i
                M[i, j] = M[j, i] += 2/Δx * dot(basis[xx, i] .* basis[xx, j], w)
            end
        end
        a = b
    end

    return  M
end

# Todo more precisely, defined on the same domain
function mass_matrix(basis1::BSplineBasis{T}, basis2::BSplineBasis{T}) where {T}
    x, w = gausslegendre(basis1.k + 2)
    a1 = basis1.interval.left
    b1 = basis1.interval.right
    N1 = length(basis1.B)
    Δx1 = (b1-a1)/basis1.N

    a2 = basis2.interval.left
    b2 = basis2.interval.right
    N2 = length(basis2.B)
    Δx2 = (b2-a2)/basis2.N
    

    M = zeros(N1, N2)
    # Do Quadrature over every element -> exact
    for el1 in 1:basis1.N
        b1 = a1 + Δx1
        xx = a1 .+ (b1 - a1) .*(x .+ 1) ./ 2

        if xx[end] > basis1.interval.right
            xx[end] = basis1.interval.right
        end

        for i in 1:N1
            for j in 1:N2
                M[i, j] += 2/(b1 - a1) .* dot(basis1[xx, i] .* basis2[xx, j], w)
            end
        end
        a1 = b1
    end

    return  M
end

# For the stiffness_matrix of basis b, call stiffness_matrix(b')
# Or should we take b as an input? 
# Care for the factor in the integral, the derived basis functions, i.e. b', have already 2/Δx as an factor!
"""
    stiffness_matrix(b'::Basis)

    Calculate the stiffnes-matrix for the basis `b` with Gauss quadrature. Caution: We directly input b' here as a basis. 
"""
function stiffness_matrix(basis::BSplineBasis{dirichlet})
    x, w = gausslegendre(basis.k + 1)
    a = basis.interval.left
    b = basis.interval.right
    N = length(basis.B)
    Δx = (b-a)/basis.N

    M = zeros(N+1, N+1)
    # The derivative of the basis functions is the known formula for BSplines
    # B'_i(x) = D_i(x) - D_{i+1}(x), where D_i are the basisfunctions of b'
    for el in 1:basis.N
        b = a + Δx
        xx = a .+ (b - a) .*(x .+ 1) ./ 2

        if xx[end] > basis.interval.right
            xx[end] = basis.interval.right
        end

        M[1, 1] += 2/(b - a) .* dot((-basis[xx, 1]) .* (-basis[xx, 1] ), w)

        for j in 1:N-1
            M[1, j+1] = M[j+1, 1] += 2/(b - a) .* dot((-basis[xx, 1]) .* (basis[xx, j] - basis[xx, j+1] ), w)
        end

        for i in 1:N-1
            for j in 1:i

                M[i+1, j+1] = M[j+1, i+1] += 2/(b - a) .* dot((basis[xx, i] .- basis[xx, i+1]) .* (basis[xx, j] .- basis[xx, j+1]), w)
            end
        end

        for j in 1:N-1
            M[N+1, j+1] = M[j+1, N+1] += 2/(b - a) .* dot((basis[xx, N]) .* (basis[xx, j] - basis[xx, j+1] ), w)
        end

        M[N+1, N+1] += 2/(b - a) .* dot((basis[xx, N]) .* (basis[xx, N] ), w)

        a = b
    end

    return  M
end

# For periodic boundary conditions on the basis use the circulant methods
function mass_matrix(basis::BSplineBasis{periodic})    
    x, w = gausslegendre(basis.k+1)
    a = basis.interval.left
    b = basis.interval.right
    N = length(basis.B)
    Δx = (b-a)/basis.N
    c = zeros(N)

    for el in 1:basis.N
        b = a + Δx
        xx = a .+ (b - a) .*(x .+ 1) ./ 2

        for i in 1:N
            c[i] += 2/(b - a) .* dot(basis[xx, i] .* basis[xx, 1], w)
        end
        a = b
    end

    return Circulant(c)  
end

# Input derivative basis
function stiffness_matrix(basis::BSplineBasis{periodic})    
    x, w = gausslegendre(basis.k+1)
    a = basis.interval.left
    b = basis.interval.right
    N = length(basis.B)
    Δx = (b-a)/basis.N
    c = zeros(N)

    for el in 1:basis.N
        b = a + Δx
        xx = a .+ (b - a) .*(x .+ 1) ./ 2

        for i in 1:N-1
            c[i] += 2/(b - a) .* dot((basis[xx, i] .- basis[xx, i+1]) .* (basis[xx, 1] .- basis[xx, 2]), w)
        end
        c[N] += 2/(b - a) .* dot((basis[xx, N] .- basis[xx, 1]) .* (basis[xx, 1] .- basis[xx, 2]), w)
        a = b
    end
    
    return Circulant(c)  
end

# Since for now we only have a periodic basis, f has to be as well
function L2_prod(f::Function, basis::Basis, order::Int64=basis.k+1)
    # what degree do we want here?
    x, w = gausslegendre(order+1)
    a = basis.interval.left
    b = basis.interval.right
    N = length(basis.B)
    Δx = (b-a)/basis.N
    fh = zeros(N)

    for el in 1:basis.N
        b = a + Δx
        xx = a .+ (b - a) .*(x .+ 1) ./ 2

        if xx[end] > basis.interval.right
            xx[end] = basis.interval.right
        end

        for i in 1:N
            fh[i] += 2/(b - a) .* dot(basis[xx, i] .* f.(xx), w)
        end
        a = b
    end

    return fh
end

#maybe find a better name
function col_mat(basis::Basis, m::Int)
    n = length(basis.B)
    x = range(basis.interval.left+1e-14, basis.interval.right-1e-14, length = m)
    M = zeros(m, n)

    for i in 1:n
        M[:,i] = basis[x, i]
    end

    return x, M
end

#n-D Calculations
function mass_matrix(basis::TensorProductBasis{N}) where N
    matrices = []
    for i in 1:N
        push!(matrices, mass_matrix(basis[i]))
    end

    return ⊗(matrices...)
end

function stiffness_matrix(basis::TensorProductBasis{N}) where N
    matrices = []
    for i in 1:N
        push!(matrices,  stiffness_matrix(basis[i]))
    end

    return ⊗(matrices...)
end

#Todo: Generalize the following functions and make them more efficient
function L2_prod(f::Function, basis::BSplineTensorProductBasis{2, T}) where T
    # what degree do we want here?
    X,Y = basis.B

    x1, w1 = gausslegendre(X.k + 1)
    x2, w2 = gausslegendre(Y.k + 1)

    a1 = X.interval.left
    b1 = X.interval.right
    N1 = length(X.B)
    Δx1 = (b1-a1)/X.N

    a2 = Y.interval.left
    b2 = Y.interval.right
    N2 = length(Y.B)
    Δx2 = (b2-a2)/Y.N

    vectors = zeros(N1, N2)

    #loop over basis functions
    for i in 1:N1
        for j in 1:N2
            #loop over elements in x-direction
            for el_x in 1:X.N
                b1 = a1 + Δx1
                xx1 = a1 .+ (b1 - a1) .*(x1 .+ 1) ./ 2

                if xx1[end] > X.interval.right
                    xx1[end] = X.interval.right
                end

                val_y = zeros(length(xx1))
                # at every quadrature point in x-direction, do a full quadrature in y-direction
                for xn in 1:length(xx1)
                    val = 0

                    for el_y in 1:Y.N
                        b2 = a2 + Δx2
                        xx2 = a2 .+ (b2 - a2) .*(x2 .+ 1) ./ 2
                        if xx2[end] > Y.interval.right
                            xx2[end] = Y.interval.right
                        end

                        val += 2/(b2 - a2) .* dot(Y[xx2, j] .* f.(xx1[xn], xx2), w2)

                        a2 = b2
                    end
                    
                    val_y[xn] = val
                    a2 = Y.interval.left
                end

                vectors[i, j] += 2/(b1 - a1) .* dot(X[xx1, i] .* val_y, w1) 
                
                a1 = b1
            end

            a1 = X.interval.left
        end

    end

    return reshape(permutedims(vectors, [2 ,1]), N1*N2)
end

function L2_prod(f::Function, basis::BSplineTensorProductBasis{3, T}) where T
    X, Y, Z = basis.B
    N = (length(X.B), length(Y.B), length(Z.B))

    x1, w1 = gausslegendre(X.k + 1)
    x2, w2 = gausslegendre(Y.k + 1)
    x3, w3 = gausslegendre(Z.k + 1)

    a1 = X.interval.left
    b1 = X.interval.right
    N1 = length(X.B)
    Δx1 = (b1-a1)/X.N

    a2 = Y.interval.left
    b2 = Y.interval.right
    N2 = length(Y.B)
    Δx2 = (b2-a2)/Y.N

    a3 = Z.interval.left
    b3 = Z.interval.right
    N3 = length(Z.B)
    Δx3 = (b3-a3)/Z.N

    vectors = zeros(Float64, N1, N2, N3)

    xx1 = zeros(Float64, length(x1))
    xx2 = zeros(Float64, length(x2))
    xx3 = zeros(Float64, length(x3))
    #loop over basis functions
    for i in 1:N1
        for j in 1:N2
            for l in 1:N3
                #loop over elements in x-direction
                for el_x in 1:X.N
                    b1 = a1 + Δx1
                    xx1 = a1 .+ (b1 - a1)/2 * (x1 .+ 1)

                    if xx1[end] > X.interval.right
                        xx1[end] = X.interval.right
                    end

                    val_y = zeros(Float64, length(xx1))
                    # at every quadrature point in x-direction, do a full quadrature in y-direction
                    for xn in 1:length(xx1)
                        val2 = 0.0

                        for el_y in 1:Y.N
                            b2 = a2 + Δx2
                            xx2 = a2 .+ (b2 - a2)/2 * (x2 .+ 1)
                            if xx2[end] > Y.interval.right
                                xx2[end] = Y.interval.right
                            end
                                val_z = zeros(Float64, length(xx2))
                                # at every quadrature point in x-direction, do a full quadrature in y-direction
                                for yn in 1:length(xx2)
                                    val3 = 0.0

                                    for el_z in 1:Z.N
                                        b3 = a3 + Δx3
                                        xx3 = a3 .+ (b3 - a3)/2 * (x3 .+ 1) 
                                        if xx3[end] > Z.interval.right
                                            xx3[end] = Z.interval.right
                                        end

                                        val3 += 2/(b3 - a3) * dot(Z[xx3, l] .* f.(xx1[xn], xx2[yn], xx3), w3)

                                        a3 = b3
                                    end
                                    
                                    val_z[yn] = val3
                                    a3 = Z.interval.left
                                end

                            val2 += 2/(b2 - a2) * dot(Y[xx2, j] .* val_z, w2)

                            a2 = b2
                        end
                        
                        val_y[xn] = val2
                        a2 = Y.interval.left
                    end
                    vectors[i, j, l] += 2/(b1 - a1) * dot(X[xx1, i] .* val_y, w1) 

                    a1 = b1
                end
                
                a1 = X.interval.left
            end
        end

    end

    return reshape(permutedims(vectors, [3, 2 ,1]), N1*N2*N3)
end

#x = (x, y)  1-d interval discretizations
function col_mat(basis::TensorProductBasis{2}, mm::Int) 
    X,Y = basis.B
    N = (length(X.B), length(Y.B))

    x = range(X.interval.left+1e-14, X.interval.right-1e-14, length = mm)
    y = range(Y.interval.left+1e-14, Y.interval.right-1e-14, length = mm)
    xy = [(a,b) for a in x for b in y]

    M1 = zeros(mm, N[1])
    M2 = zeros(mm, N[2])

    for i in 1:N[1]
        M1[:, i] = X[x, i]
    end

    for i in 1:N[2]
        M2[:, i] = Y[y, i]
    end

    return xy, M1 ⊗ M2
end

function col_mat(basis::BSplineTensorProductBasis{3, T}, mm::Int) where T
    X, Y, Z = basis.B
    N = (length(X.B), length(Y.B), length(Z.B))

    x = range(X.interval.left+1e-14, X.interval.right-1e-14, length = mm)
    y = range(Y.interval.left+1e-14, Y.interval.right-1e-14, length = mm)
    z = range(Z.interval.left+1e-14, Z.interval.right-1e-14, length = mm)
    xyz = [(a, b, c) for a in x for b in y for c in z]    
        
    M1 = zeros(mm, N[1])
    M2 = zeros(mm, N[2])
    M3 = zeros(mm, N[3])

    for i in 1:N[1]
        M1[:, i] = X[x, i]
    end

    for i in 1:N[2]
        M2[:, i] = Y[y, i]
    end

    for i in 1:N[3]
        M3[:, i] = Z[z, i]
    end

    return xyz, M1 ⊗ M2 ⊗ M3
end

function L2_prod(f::Function, basis::SArray{Tuple{3}, BSplineTensorProductBasis{3, T}}) where {T}
    fh = []

    for i in 1:3
        push!(fh, L2_prod((x,y,z) -> f(x,y,z)[i], basis[i]))
    end

    return SVector(fh...)
end

function L2_prod(f::Function, basis::SArray{Tuple{2}, BSplineTensorProductBasis{2, T}}) where {T}
    fh = []

    for i in 1:2
        push!(fh, L2_prod((x,y) -> f(x,y)[i], basis[i]))
    end

    return SVector(fh...)
end

function L2_proj(f::Function, basis::SArray{Tuple{3}, BSplineTensorProductBasis{3, T}}) where {T}
    fh = []
    M = mass_matrix.(basis)

    for i in 1:3
        push!(fh, M[i]\L2_prod((x,y,z) -> f(x,y,z)[i], basis[i]))
    end

    return SVector(fh...)
end

function L2_proj(f::Function, basis::BSplineTensorProductBasis{3, T}) where {T}
    M = mass_matrix(basis)
    
    fh = M\L2_prod(f, basis)
    return SVector{length(fh)}(fh)
end

function L2_proj(f::Function, basis::SArray{Tuple{2}, BSplineTensorProductBasis{2, T}}) where {T}
    fh = []
    M = mass_matrix.(basis)

    for i in 1:2
        push!(fh, M[i]\L2_prod((x,y) -> f(x,y)[i], basis[i]))
    end

    return SVector(fh...)
end

function L2_proj(f::Function, basis::BSplineTensorProductBasis{2, T}) where {T}
    M = mass_matrix(basis)
    fh = M\L2_prod(f, basis)
    return SVector{length(fh)}(fh)
end

function L2_proj(f::Function, basis::BSplineBasis{T}) where {T}
    M = mass_matrix(basis)
    fh = M\L2_prod(f, basis)
    return SVector{length(fh)}(fh)
end

##weighted mass matrices
function weighted_mass_matrix(basis::Basis, ω::Function)
    x, w = gausslegendre(basis.k + 1)
    a = basis.interval.left
    b = basis.interval.right
    N = length(basis.B)
    Δx = (b-a)/basis.N

    M = zeros(N, N)

    for el in 1:basis.N
        b = a + Δx
        xx = a .+ (b - a) .*(x .+ 1) ./ 2
        if xx[end] > basis.interval.right
            xx[end] = basis.interval.right
        end

        for i in 1:N
            for j in 1:N
                M[i, j] += 2/(b - a) .* dot(basis[xx, i] .* basis[xx, j] .* ω.(xx), w)
            end
        end
        a = b
    end

    return  M
end

#ω(x,y)
function weighted_mass_matrix(basis::BSplineTensorProductBasis{2, T}, ω::Function) where {T}
    X,Y = basis.B

    #maybe we need higher order
    x1, w1 = gausslegendre(X.k + 1)
    x2, w2 = gausslegendre(Y.k + 1)

    a1 = X.interval.left
    b1 = X.interval.right
    N1 = length(X.B)
    Δx1 = (b1-a1)/X.N

    a2 = Y.interval.left
    b2 = Y.interval.right
    N2 = length(Y.B)
    Δx2 = (b2-a2)/Y.N

    M = zeros(N1*N2, N1*N2)
    
    vectors = zeros(N1, N2)
    for k in 1:N1
        for l in 1:N2
            
            vectors = zeros(N1, N2)
            for i in 1:N1
                for j in 1:N2
                    #loop over elements in x-direction
                    for el_x in 1:X.N
                        b1 = a1 + Δx1
                        xx1 = a1 .+ (b1 - a1) .*(x1 .+ 1) ./ 2
                        if xx1[end] > X.interval.right
                            xx1[end] = X.interval.right
                        end

                        val_y = zeros(length(xx1))
                        # at every quadrature point in x-direction, do a full quadrature in y-direction
                        for xn in 1:length(xx1)
                            val = 0

                            for el_y in 1:Y.N
                                b2 = a2 + Δx2
                                xx2 = a2 .+ (b2 - a2) .*(x2 .+ 1) ./ 2
                                if xx2[end] > Y.interval.right
                                    xx2[end] = Y.interval.right
                                end
                                val += 2/(b2 - a2) .* dot(Y[xx2, j] .* Y[xx2, l] .* ω.(xx1[xn], xx2), w2)

                                a2 = b2
                            end

                            val_y[xn] = val
                            a2 = Y.interval.left
                        end
                        vectors[i, j] += 2/(b1 - a1) .* dot(X[xx1, i] .* X[xx1, k] .* val_y, w1) 

                        a1 = b1
                    end
                    a1 = X.interval.left
                end

            end
            M[:, N2*(k-1) + l ] = reshape(permutedims(vectors, [2 ,1]), N1*N2)
            
        end
    end

    return  M
end

#special structure because of V1
#w(x, y)
function interior_product_matrix(basis::SArray{Tuple{2}, BSplineTensorProductBasis{2, T}}, ω::Function) where T
    M = weighted_mass_matrix(basis[1], basis[2], ω)
    
    return SArray(M, M')
end

#ω(x,y)
#V1 = SVector(X' ⊗ Y, X ⊗ Y')
function weighted_mass_matrix(basis1::BSplineTensorProductBasis{2, periodic}, basis2::BSplineTensorProductBasis{2, periodic}, ω::Function) 
    X, Y = basis1.B 
    X2, Y2 = basis2.B 
    
    x1, w1 = gausslegendre(X.k + 1)
    x2, w2 = gausslegendre(Y.k + 1)

    # Assume both component bases are defined on the same domain
    a1 = X.interval.left
    b1 = X.interval.right
    N1 = length(X.B)
    Δx1 = (b1-a1)/X.N

    a2 = Y.interval.left
    b2 = Y.interval.right
    N2 = length(Y.B)
    Δx2 = (b2-a2)/Y.N

    M = zeros(N1*N2, N1*N2)
    
    vectors = zeros(N1, N2)
    for k in 1:length(X2.B) # build the full matrix 
        for l in 1:length(Y2.B)
            
            vectors = zeros(N1, N2)
            for i in 1:N1
                for j in 1:N2
                    #loop over elements in x-direction
                    for el_x in 1:X.N
                        b1 = a1 + Δx1
                        xx1 = a1 .+ (b1 - a1) .*(x1 .+ 1) ./ 2
                        if xx1[end] > X.interval.right
                            xx1[end] = X.interval.right
                        end
                        val_y = zeros(length(xx1))
                        # at every quadrature point in x-direction, do a full quadrature in y-direction
                        for xn in 1:length(xx1)
                            val = 0

                            for el_y in 1:Y.N
                                b2 = a2 + Δx2
                                xx2 = a2 .+ (b2 - a2) .*(x2 .+ 1) ./ 2
                                if xx2[end] > Y.interval.right
                                    xx2[end] = Y.interval.right
                                end
                                val += 2/(b2 - a2) .* dot(Y[xx2, j] .* Y2[xx2, l] .* ω.(xx1[xn], xx2), w2)

                                a2 = b2
                            end

                            val_y[xn] = val
                            a2 = Y.interval.left
                        end
                        vectors[i, j] += 2/(b1 - a1) .* dot(X[xx1, i] .* X2[xx1, k] .* val_y, w1) 

                        a1 = b1
                    end
                    a1 = X.interval.left
                end

            end
            M[:, N2*(k-1) + l ] = reshape(permutedims(vectors, [2 ,1]), N1*N2)
            
        end
    end

    return  M
end

function interior_product_matrix(basis::SArray{Tuple{3}, BSplineTensorProductBasis{3, T}}, ω::SArray{Tuple{3}, Function}) where T
    
    M1 = weighted_mass_matrix(V1[1], V1[2], ω[3])
    M2 = weighted_mass_matrix(V1[1], V1[3], ω[2])
    M3 = weighted_mass_matrix(V1[2], V1[3], ω[1])
    
    #M = [0 -M1 M2;
    #     M1 0 -M3;
    #     -M2 M3 0]

    #we could take it as a curl matrix
    return SVector(M1, M2, M3)
end

function L2_penalty(uh::T, basis::BSplineBasis, ϵ) where T

    M = mass_matrix(basis)
    M0 = copy(M)

    M0[1, 1] *= (1 + 1/ϵ) 
    M0[end, end] *= (1 + 1/ϵ)

    return M0\(M*uh)

end

#returns invertible matrix, to get coefficients from point values
function col_mat_proj(basis::BSplineTensorProductBasis{2, periodic}) 
    X,Y = basis.B
    N = (length(X.B), length(Y.B))
    k = (X.k, Y.k)

    #periodic
    x = X.t[k[1]+2:N[1]+k[1]+1]
    y = Y.t[k[2]+2:N[2]+k[2]+1]

    xy = [(a,b) for a in x for b in y]

    M1 = zeros(N[1], N[1])
    M2 = zeros(N[2], N[2])

    for i in 1:N[1]
        M1[:, i] = X[x, i]
    end

    for i in 1:N[2]
        M2[:, i] = Y[y, i]
    end

    return xy, M1 ⊗ M2
end

#returns invertible matrix, to get coefficients from point values
function col_mat_proj(basis::BSplineTensorProductBasis{2, dirichlet}) 
    X,Y = basis.B
    N = (length(X.B), length(Y.B))
    k = (X.k, Y.k)

    
    x = range(X.interval.left+1e-14, X.interval.right-1e-14, length= N[1])
    y = range(Y.interval.left+1e-14, Y.interval.right-1e-14, length= N[2])
    xy = [(a,b) for a in x for b in y]

    M1 = zeros(N[1], N[1])
    M2 = zeros(N[2], N[2])

    for i in 1:N[1]
        M1[:, i] = X[x, i]
    end

    for i in 1:N[2]
        M2[:, i] = Y[y, i]
    end

    return xy, M1 ⊗ M2
end

#returns invertible matrix, to get coefficients from point values
function col_mat_proj(basis::BSplineBasis{dirichlet}) 
    N = length(basis.B)
    #k = basis.k
    
    x = range(basis.interval.left+1e-14, basis.interval.right-1e-14, length= N)

    M1 = zeros(N, N)

    for i in 1:N
        M1[:, i] = basis[x, i]
    end

    return x, M1 
end