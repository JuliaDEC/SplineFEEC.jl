# Name util.jl is a bit weird. Maybe something like metric.jl is better, since we mainly calculate integrals here.
# 1-D Calculations
function mass_matrix(basis::Basis)
    x, w = gausslobatto(basis.k + 1)
    a = basis.interval.left
    b = basis.interval.right
    N = length(basis.B)
    Δx = (b-a)/N

    M = zeros(N, N)
    # Do Quadrature over every element -> exact
    for el in 1:N
        b = a + Δx
        xx = a .+ (b - a) .*(x .+ 1) ./ 2
        for i in 1:N
            for j in 1:i
                M[i, j] = M[j, i] += 2/(b - a) .* dot(basis[xx, i] .* basis[xx, j], w)
            end
        end
        a = b
    end

    return  M
end

# Todo more precisely
function mass_matrix(basis1::Basis, basis2::Basis)
    x, w = gausslobatto(basis1.k + 1)
    a = basis1.interval.left
    b = basis1.interval.right
    N = length(basis1.B)
    Δx = (b-a)/N

    M = zeros(N, N)
    # Do Quadrature over every element -> exact
    for el in 1:N
        b = a + Δx
        xx = a .+ (b - a) .*(x .+ 1) ./ 2
        for i in 1:N
            for j in 1:N
                M[i, j] += 2/(b - a) .* dot(basis1[xx, i] .* basis2[xx, j], w)
            end
        end
        a = b
    end

    return  M
end

# For the stiffness_matrix of basis b, call stiffness_matrix(b')
# Or should we take b as an input? 
# Care for the factor in the integral, the derived basis functions, i.e. b', have already 2/Δx as an factor!
function stiffness_matrix(basis::Basis)
    x, w = gausslobatto(basis.k + 1)
    a = basis.interval.left
    b = basis.interval.right
    N = length(basis.B)
    Δx = (b-a)/N

    M = zeros(N, N)
    # The derivative of the basis functions is the known formula for BSplines
    # B'_i(x) = D_i(x) - D_{i+1}(x), where D_i are the basisfunctions of b'
    for el in 1:N
        b = a + Δx
        xx = a .+ (b - a) .*(x .+ 1) ./ 2
        for i in 1:N-1
            for j in 1:i
                M[i, j] = M[j, i] += 2/(b - a) .* dot((basis[xx, i] .- basis[xx, i+1]) .* (basis[xx, j] .- basis[xx, j+1]), w)
            end
        end

        for j in 1:N-1
            M[N, j] = M[j, N] += 2/(b - a) .* dot((basis[xx, N] .- basis[xx, 1]) .* (basis[xx, j] .- basis[xx, j + 1]), w)
        end

        M[N, N] += 2/(b - a) .* dot((basis[xx, N] .- basis[xx, 1]) .* (basis[xx, N] .- basis[xx, 1]), w)

        a = b
    end

    return  M
end

# For periodic boundary conditions on the basis use the circulant methods
function mass_matrix(basis::BSplineBasis{periodic})    
    x, w = gausslobatto(basis.k+1)
    a = basis.interval.left
    b = basis.interval.right
    N = length(basis.B)
    Δx = (b-a)/N
    c = zeros(N)

    for el in 1:N
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
# Does not work for order k=2, because the periodic function value of one basis function is "broken" at the right edge
function stiffness_matrix(basis::BSplineBasis{periodic})    
    x, w = gausslobatto(basis.k+1)
    a = basis.interval.left
    b = basis.interval.right
    N = length(basis.B)
    Δx = (b-a)/N
    c = zeros(N)

    for el in 1:N
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
function L2_prod(f::Function, basis::Basis)
    # what degree do we want here?
    x, w = gausslobatto(basis.k + 1)
    a = basis.interval.left
    b = basis.interval.right
    N = length(basis.B)
    Δx = (b-a)/N
    fh = zeros(N)

    for el in 1:N
        b = a + Δx
        xx = a .+ (b - a) .*(x .+ 1) ./ 2

        for i in 1:N
            fh[i] += 2/(b - a) .* dot(basis[xx, i] .* f.(xx), w)
        end
        a = b
    end

    return fh
end

function col_mat(basis::Basis, m::Int)
    n = length(basis.B)
    x = range(basis.interval.left, basis.interval.right, length = m)
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

function stiffness_matrix_circulant(basis::TensorProductBasis{N}) where N
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

    x1, w1 = gausslobatto(X.k + 1)
    x2, w2 = gausslobatto(Y.k + 1)

    a1 = X.interval.left
    b1 = X.interval.right
    N1 = length(X.B)
    Δx1 = (b1-a1)/N1

    a2 = Y.interval.left
    b2 = Y.interval.right
    N2 = length(Y.B)
    Δx2 = (b2-a2)/N2

    vectors = zeros(N1, N2)

    #loop over basis functions
    for i in 1:N1
        for j in 1:N2
            #loop over elements in x-direction
            for el_x in 1:N1
                b1 = a1 + Δx1
                xx1 = a1 .+ (b1 - a1) .*(x1 .+ 1) ./ 2

                val_y = zeros(length(xx1))
                # at every quadrature point in x-direction, do a full quadrature in y-direction
                for xn in 1:length(xx1)
                    val = 0

                    for el_y in 1:N2
                        b2 = a2 + Δx2
                        xx2 = a2 .+ (b2 - a2) .*(x2 .+ 1) ./ 2

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

    x1, w1 = gausslobatto(X.k + 1)
    x2, w2 = gausslobatto(Y.k + 1)
    x3, w3 = gausslobatto(Z.k + 1)

    a1 = X.interval.left
    b1 = X.interval.right
    N1 = length(X.B)
    Δx1 = (b1-a1)/N1

    a2 = Y.interval.left
    b2 = Y.interval.right
    N2 = length(Y.B)
    Δx2 = (b2-a2)/N2

    a3 = Z.interval.left
    b3 = Z.interval.right
    N3 = length(Z.B)
    Δx3 = (b3-a3)/N3

    vectors = zeros(Float64, N1, N2, N3)

    xx1 = zeros(Float64, length(x1))
    xx2 = zeros(Float64, length(x2))
    xx3 = zeros(Float64, length(x3))
    #loop over basis functions
    for i in 1:N1
        for j in 1:N2
            for l in 1:N3
                #loop over elements in x-direction
                for el_x in 1:N1
                    b1 = a1 + Δx1
                    xx1 = a1 .+ (b1 - a1)/2 * (x1 .+ 1)


                    val_y = zeros(Float64, length(xx1))
                    # at every quadrature point in x-direction, do a full quadrature in y-direction
                    for xn in 1:length(xx1)
                        val2 = 0.0

                        for el_y in 1:N2
                            b2 = a2 + Δx2
                            xx2 = a2 .+ (b2 - a2)/2 * (x2 .+ 1)

                                val_z = zeros(Float64, length(xx2))
                                # at every quadrature point in x-direction, do a full quadrature in y-direction
                                for yn in 1:length(xx2)
                                    val3 = 0.0

                                    for el_z in 1:N3
                                        b3 = a3 + Δx3
                                        xx3 = a3 .+ (b3 - a3)/2 * (x3 .+ 1) 

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

    x = range(X.interval.left, X.interval.right, length = mm)
    y = range(Y.interval.left, Y.interval.right, length = mm)
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

    x = range(X.interval.left, X.interval.right, length = mm)
    y = range(Y.interval.left, Y.interval.right, length = mm)
    z = range(Z.interval.left, Z.interval.right, length = mm)
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
    
    fh = M\L2_prod((x,y,z) -> f(x,y,z), basis)
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
    fh = M\L2_prod((x,y) -> f(x,y), basis)
    return SVector{length(fh)}(fh)
end