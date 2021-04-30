struct MultipatchBSplineBasis{T} <: Basis where {T}
    B::BSplineBasis #Defined on interval [0,1]
    interval::AbstractInterval #Domain
    el::Int #Number of elements in the domain 
end

function DirichletMultipatchBSpline(interval::AbstractInterval, k::Int, N::Int, el::Int)
    B = DirichletBSpline(0..1, k, N)

    return MultipatchBSplineBasis{dirichlet}(B, interval, el)
end

function derivative(basis::MultipatchBSplineBasis{T}) where {T}

    return MultipatchBSplineBasis{T}(basis.B', basis.interval, basis.el)
end

function adjoint(basis::MultipatchBSplineBasis{T}) where {T}
 
    return derivative(basis)
end

function L2_prod(f::Function, basis::MultipatchBSplineBasis{dirichlet})
    a = basis.interval.left
    b = basis.interval.right
    Δx = (b-a)/basis.el
    n = length(basis.B.B)
    uh = zeros(n, basis.el)

    for i in 1:basis.el
        g = x -> f( a + Δx * x  )
        uh[:, i] = 1/Δx * L2_prod(g, basis.B) 
        a += Δx 
    end
    
    # Which representation is better? 
    uh2 = zeros((n-1)*basis.el+1)
    for i in 1:basis.el
        uh2[(i-1)*(n-1)+1:i*(n-1)+1] += uh[1:end, i]
    end

    return uh2
end

function L2_prod_d(f::Function, basis::MultipatchBSplineBasis{dirichlet})
    a = basis.interval.left
    b = basis.interval.right
    Δx = (b-a)/basis.el
    n = length(basis.B.B)
    uh = zeros(n, basis.el)

    for i in 1:basis.el
        g = x -> f( a + Δx * x  )
        uh[:, i] = 1/Δx^2 * L2_prod(g, basis.B) 
        a += Δx 
    end
    
    # Which representation is better? 
    uh2 = zeros(n*basis.el)
    for i in 1:basis.el
        uh2[(i-1)*n+1:i*n] += uh[1:end, i]
    end

    return uh2
end

function mass_matrix(basis::MultipatchBSplineBasis{dirichlet})
    
    a = basis.interval.left
    b = basis.interval.right
    Δx = (b-a)/basis.el
    n = length(basis.B.B)

    M = mass_matrix(basis.B)

    MM = zeros((n-1)*basis.el+1, (n-1)*basis.el+1)
    for i in 1:basis.el
        MM[(i-1)*(n-1)+1:i*(n-1)+1, (i-1)*(n-1)+1:i*(n-1)+1] += M
    end

    return 1/Δx .* MM
end

function mass_matrix_d(basis::MultipatchBSplineBasis{dirichlet})
    
    a = basis.interval.left
    b = basis.interval.right
    Δx = (b-a)/basis.el
    n = length(basis.B.B)

    M = mass_matrix(basis.B)

    MM = zeros(n*basis.el, n*basis.el)
    for i in 1:basis.el
        MM[(i-1)*n+1:i*n, (i-1)*n+1:i*n] += M
    end

    return 1/Δx^3 .* MM
end

function stiffness_matrix(basis::MultipatchBSplineBasis{dirichlet})
    a = basis.interval.left
    b = basis.interval.right
    Δx = (b-a)/basis.el
    n = length(basis.B.B)

    M = stiffness_matrix(basis.B)
    MM = zeros(n*basis.el+1, n*basis.el+1)

    for i in 1:basis.el
        MM[(i-1)*n+1:i*(n)+1, (i-1)*n+1:i*n+1] += M
    end

    return 1/Δx^3 .* MM
end

function L2_proj(f::Function, basis::MultipatchBSplineBasis{T}) where {T}
    M = mass_matrix(basis)

    return M\L2_prod(f, basis)
end

function L2_proj_d(f::Function, basis::MultipatchBSplineBasis{T}) where {T}
    M = mass_matrix_d(basis)

    return M\L2_prod_d(f, basis)
end

function der_mat(basis::MultipatchBSplineBasis{dirichlet})
    el = basis.el
    n = length(basis'.B.B)
    n2 = length(basis.B.B)
    DD = zeros(n*basis.el, (n2-1)*basis.el+1)
    D = der_mat(n2)[2:end, :]

    # n-1 x n matrix 
    for i in 1:el
        DD[1 + (i-1)*n:i*n,1 + (i-1)*(n2-1):i*(n2-1)+1] += D
    end

    DD[end, end] = 1
    return DD
end

function der_mat_d(basis::MultipatchBSplineBasis{dirichlet})
    el = basis.el
    n = length(basis'.B.B)
    n2 = length(basis.B.B)
    DD = zeros(n*basis.el, n2*basis.el)
    D = der_mat(n2)[2:end, :]

    for i in 1:el
        DD[1 + (i-1)*n:i*n,1 + (i-1)*n2:i*n2] += D
    end

    return DD
end

function disc_grad(basis::MultipatchBSplineBasis{T}) where {T}
    return der_mat(basis)
end

function col_mat(basis::MultipatchBSplineBasis{dirichlet}, m::Int)
    a = basis.interval.left
    b = basis.interval.right
    Δx = (b-a)/basis.el
    n = length(basis.B.B)
    k = basis.B.k
    N = basis.B.N
    el = basis.el

    CM = zeros((m-1) * el+1, (n-1)*el+1)
    cm = col_mat(basis.B, m)[2]

    for i in 1:el
        CM[(i-1)*(m-1)+1:i*(m-1)+1, (i-1)*(n-1)+1:i*(n-1)+1] = cm
    end

    xx = range(a, b, length = (m-1)*el+1)
    return xx, CM
end

function col_mat_d(basis::MultipatchBSplineBasis{dirichlet}, m::Int)
    a = basis.interval.left
    b = basis.interval.right
    Δx = (b-a)/basis.el
    n = length(basis.B.B)
    k = basis.B.k
    N = basis.B.N
    el = basis.el

    CM = zeros(m*el, n*el)
    cm = col_mat(basis.B, m)[2]

    for i in 1:el
        CM[(i-1)*m+1:i*m, (i-1)*(n)+1:i*n]  = 1/Δx * cm
    end

    xx = range(0, Δx, length = m)
    x = zeros(m*el)

    for i in 1:el
        x[(i-1)*m + 1:i*m] += a .+ Δx * (i-1) .+ xx
    end

    return x, CM
end

function conforming_averaging_projection(c::Vector, basis::MultipatchBSplineBasis{dirichlet})
    n = length(basis.B.B)
    el = basis.el
    
    new_c = zeros(el *(n-1)+1)
    for i in 1:el
        new_c[1+(i-1)*(n-1):i*(n-1)+1] += c[1+(i-1)*(n-1)+(i-1):i*(n-1)+(i-1)+1]
    end
    
    new_c[(n-1)+1:(n-1):(el-1)*(n-1)+1] *= 1/2
    
    return new_c
end

function conforming_averaging_projection_matrix(basis::MultipatchBSplineBasis{dirichlet})
    n = length(basis.B.B)
    el = basis.el
    
    new_c = zeros(el *(n-1)+1)
    
    M = zeros(el*(n-1)+1, el*n)
    for i in 1:el
        M[1+(i-1)*(n-1):i*(n-1)+1, 1+(i-1)*(n-1)+(i-1):i*(n-1)+1+(i-1)] += I
    end
    
    M[(n-1)+1:(n-1):(el-1)*(n-1)+1, :] *= 1/2
    
    return M
end

function weighted_mass_matrix(basis::MultipatchBSplineBasis{dirichlet}, ω::Function)
    a = basis.interval.left
    b = basis.interval.right
    Δx = (b-a)/basis.el
    n = length(basis.B.B)

    M = zeros((n-1)*basis.el+1, (n-1)*basis.el+1)


    for i in 1:basis.el
        g = x -> ω(a + Δx * x)
        M[(i-1)*(n-1)+1:i*(n-1)+1, (i-1)*(n-1)+1:i*(n-1)+1] = 1/Δx * weighted_mass_matrix(basis.B, g) 
        a += Δx 
    end

    return M
end

function weighted_mass_matrix_d(basis::MultipatchBSplineBasis{dirichlet}, ω::Function)
    a = basis.interval.left
    b = basis.interval.right
    Δx = (b-a)/basis.el
    n = length(basis.B.B)

    M = zeros(n*basis.el, n*basis.el)


    for i in 1:basis.el
        g = x -> ω(a + Δx * x)
        M[(i-1)*n+1:i*n, (i-1)*n+1:i*n] += 1/Δx^2 * weighted_mass_matrix(basis.B, g) 
        a += Δx 
    end

    return M
end

#-----

function PeriodicMultipatchBSpline(interval::AbstractInterval, k::Int, N::Int, el::Int)
    B = DirichletBSpline(0..1, k, N)

    return MultipatchBSplineBasis{periodic}(B, interval, el)
end

function ToDirichlet(basis::MultipatchBSplineBasis{periodic})

    return MultipatchBSplineBasis{dirichlet}(basis.B, basis.interval, basis.el)
end

function L2_prod(f::Function, basis::MultipatchBSplineBasis{periodic})
    
    fh = L2_prod(f, ToDirichlet(basis))
    fh_p = fh[1:end-1]
    fh_p[1] += fh[end]

    return fh_p
end

function L2_prod_d(f::Function, basis::MultipatchBSplineBasis{periodic})
    fh = L2_prod_d(f, ToDirichlet(basis))

    return fh
end

function mass_matrix(basis::MultipatchBSplineBasis{periodic})
    
    M = mass_matrix(ToDirichlet(basis))
    M_p = zero(M[1:end-1, 1:end-1])
    M_p += M[1:end-1, 1:end-1]
    M_p[1, 1:end] += M[end, 1:end-1]
    M_p[1:end, 1] += M[1:end-1, end]
    M_p[1,1] += M[end,end]

    return M_p
end

function mass_matrix_d(basis::MultipatchBSplineBasis{periodic})
    
    M = mass_matrix_d(ToDirichlet(basis))

    return M
end

function stiffness_matrix(basis::MultipatchBSplineBasis{periodic})

    M = stiffness_matrix(ToDirichlet(basis))
    M_p = zero(M[1:end-1, 1:end-1])
    M_p += M[1:end-1, 1:end-1]
    M_p[1, 1:end] += M[end, 1:end-1]
    M_p[1:end, 1] += M[1:end-1, end]
    M_p[1,1] += M[end,end]

    return M_p

end

function col_mat(basis::MultipatchBSplineBasis{periodic}, m::Int)
    xx, cm = col_mat(ToDirichlet(basis), m)

    cm_p = zero(cm[:, 1:end-1])
    cm_p += cm[:, 1:end-1]
    cm_p[:, 1] += cm[:, end]

    return xx, cm_p
end

function col_mat_d(basis::MultipatchBSplineBasis{periodic}, m::Int)
    xx, cm = col_mat_d(ToDirichlet(basis), m)

    return xx, cm
end

#this is not fully correct I think, as I don't know how the derivative of the periodic function should behave...
function der_mat(basis::MultipatchBSplineBasis{periodic})
    DD = SplineFEEC.der_mat(SplineFEEC.ToDirichlet(basis))
    D_p = zero(DD[ :, 1:end-1])
    D_p[ 1:end-1, 1:end] += DD[ 1:end-1, 1:end-1]
    D_p[end, end] = -1
    D_p[end, 1] = 1

    return D_p
end

function der_mat_d(basis::MultipatchBSplineBasis{periodic})
    D = der_mat_d(ToDirichlet(basis))

    return D
end

function conforming_averaging_projection_matrix(basis::MultipatchBSplineBasis{periodic})
    P = conforming_averaging_projection_matrix(ToDirichlet(basis))
    P = P[1:end-1, :]
    P[1, end] = 1
    P[1, :] *= 1/2
    
    return P
end

function weighted_mass_matrix(basis::MultipatchBSplineBasis{periodic}, ω::Function)
    M = weighted_mass_matrix(ToDirichlet(basis))
    M_p = zero(M[1:end-1, 1:end-1])
    M_p += M[1:end-1, 1:end-1]
    M_p[1, 1:end] += M[end, 1:end-1]
    M_p[1:end, 1] += M[1:end-1, end]
    M_p[1,1] += M[end,end]

    return M_p
end

function weighted_mass_matrix_d(basis::MultipatchBSplineBasis{periodic}, ω::Function)
     M = weighted_mass_matrix_d(ToDirichlet(basis))

    return M
end

#-----

struct MultipatchBSplineTensorProductBasis{N, T} <: TensorProductBasis{N}
    B::SArray{Tuple{N}, MultipatchBSplineBasis{T}}
end

function DirichletMultipatchTensorProductBSpline(domain::ProductDomain, k::NTuple{d,Int}, N::NTuple{d,Int}, el::NTuple{d,Int}) where {d}
    basis =[]
    for i in 1:d
        push!(basis, DirichletMultipatchBSpline(domain.domains[i], k[i], N[i], el[i]))
    end

    return MultipatchBSplineTensorProductBasis{d, dirichlet}(SVector(basis...))
end

function PeriodicMultipatchTensorProductBSpline(domain::ProductDomain, k::NTuple{d,Int}, N::NTuple{d,Int}, el::NTuple{d,Int}) where {d}
    basis =[]
    for i in 1:d
        push!(basis, PeriodicMultipatchBSpline(domain.domains[i], k[i], N[i], el[i]))
    end

    return MultipatchBSplineTensorProductBasis{d, periodic}(SVector(basis...))
end

function ToDirichlet(basis::MultipatchBSplineTensorProductBasis{d, periodic}) where {d}

    return MultipatchBSplineTensorProductBasis{d, dirichlet}(ToDirichlet.(basis.B))
end

getindex(B::MultipatchBSplineTensorProductBasis{N, T}, j::Integer) where {N, T} = B.B[j]
getindex(B::MultipatchBSplineTensorProductBasis{N, T}, j::Colon) where {N, T} = B.B

#tensorproduct
function tensorproduct(a::MultipatchBSplineBasis{T}, b::MultipatchBSplineBasis{T}) where T
    Bs = SVector(a, b)
    MultipatchBSplineTensorProductBasis{2, T}(SVector(a, b))
end

function tensorproduct(a::MultipatchBSplineTensorProductBasis{d, T}, b::MultipatchBSplineBasis{T}) where {d, T}
    B = SVector(a.B..., b)
    MultipatchBSplineTensorProductBasis{d+1, T}(B)
end

function tensorproduct(a::MultipatchBSplineBasis{T}, b::MultipatchBSplineTensorProductBasis{d, T}) where {d, T}
    B = SVector(a, b.B...)
    MultipatchBSplineTensorProductBasis{d+1, T}(B)
end

function tensorproduct(a::MultipatchBSplineTensorProductBasis{d1, T}, b::MultipatchBSplineTensorProductBasis{d2, T}) where {d1, d2, T}
    B = SVector(a.B..., b.B...)
    MultipatchBSplineTensorProductBasis{d1+d2, T}(B)
end

⊗(a::Union{MultipatchBSplineTensorProductBasis, MultipatchBSplineBasis}, b::Union{MultipatchBSplineTensorProductBasis, MultipatchBSplineBasis}) = tensorproduct(a, b)

function derivative(basis::MultipatchBSplineTensorProductBasis{d, T}) where {d, T}
    b = derivative(basis.B[1])
    for i in 2:d
        bn = derivative(basis.B[i])
        b = b ⊗ bn
    end

    return b
end

function adjoint(basis::MultipatchBSplineTensorProductBasis)
    return derivative(basis)
end

#Methods
function L2_prod(f::Function, basis::MultipatchBSplineTensorProductBasis{2, dirichlet})
    a1 = basis[1].interval.left
    b1 = basis[1].interval.right
    el1 = basis[1].el
    Δx1 = (b1-a1)/basis[1].el
    n1 = length(basis[1].B.B)

    a2 = basis[2].interval.left
    b2 = basis[2].interval.right
    el2 = basis[2].el
    Δx2 = (b2-a2)/el2
    n2 = length(basis[2].B.B)

    uh = zeros(((n1-1)*el1+1)*((n2-1)*el2+1))

    for i in 1:el1
        for j in 1:el2
            g = (x, y) -> f(a1 + Δx1 * x, a2 + Δx2*y)
            val = 1/(Δx1 * Δx2) * L2_prod(g, basis[1].B ⊗ basis[2].B) 

            for k in 1:n1
                uh[1+(k-1)*n2+(k-1)*(el2-1)*(n2-1)+(j-1)*el2*(n2-1)+(i-1)*(el2-1)*(n2-1)*(n1-1)+(i-1)*n2*(n1-1)-(j-1)*(n2-1)*(el2-1):k*n2+(k-1)*(el2-1)*(n2-1)+(j-1)*el2*(n2-1)+(i-1)*(el2-1)*(n1-1)*(n2-1)-(j-1)*(n2-1)*(el2-1)+(i-1)*n2*(n1-1)] += val[1+(k-1)*n2:k*n2]
            end

            a2 += Δx2 
        end

        a1 += Δx1
    end
    return uh
end
 
function L2_prod_mixed_x(f::Function, basis::MultipatchBSplineTensorProductBasis{2, dirichlet})
    a1 = basis[1].interval.left
    b1 = basis[1].interval.right
    el1 = basis[1].el
    Δx1 = (b1-a1)/basis[1].el
    n1 = length(basis[1].B.B)

    a2 = basis[2].interval.left
    b2 = basis[2].interval.right
    el2 = basis[2].el
    Δx2 = (b2-a2)/el2
    n2 = length(basis[2].B.B)

    uh = zeros((n1*el1)*((n2-1)*el2+1))

    for i in 1:el1
        for j in 1:el2
            g = (x, y) -> f(a1 + Δx1 * x, a2 + Δx2*y)
            val = 1/(Δx1^2 * Δx2) * L2_prod(g, basis[1].B ⊗ basis[2].B) 

            for k in 1:n1
                uh[1+(k-1)*n2+(k-1)*(el2-1)*(n2-1)+(j-1)*el2*(n2-1)+(i-1)*(el2-1)*(n2-1)*n1+(i-1)*n2*n1-(j-1)*(n2-1)*(el2-1):k*n2+(k-1)*(el2-1)*(n2-1)+(j-1)*el2*(n2-1)+(i-1)*(el2-1)*n1*(n2-1)-(j-1)*(n2-1)*(el2-1)+(i-1)*n2*n1] += val[1+(k-1)*n2:k*n2]
            end

            a2 += Δx2 
        end

        a1 += Δx1
    end
    return uh
end

function L2_prod_mixed_y(f::Function, basis::MultipatchBSplineTensorProductBasis{2, dirichlet})
    a1 = basis[1].interval.left
    b1 = basis[1].interval.right
    el1 = basis[1].el
    Δx1 = (b1-a1)/basis[1].el
    n1 = length(basis[1].B.B)

    a2 = basis[2].interval.left
    b2 = basis[2].interval.right
    el2 = basis[2].el
    Δx2 = (b2-a2)/el2
    n2 = length(basis[2].B.B)

    uh = zeros(((n1-1)*el1+1)*n2*el2)

    for i in 1:el1
        for j in 1:el2
            g = (x, y) -> f(a1 + Δx1 * x, a2 + Δx2*y)
            val = 1/(Δx1 * Δx2^2) * L2_prod(g, basis[1].B ⊗ basis[2].B) 

            for k in 1:n1
                uh[1+(k-1)*n2+(k-1)*(el2-1)*n2+(j-1)*el2*n2+(i-1)*(el2-1)*n2*(n1-1)+(i-1)*n2*(n1-1)-(j-1)*n2*(el2-1):k*n2+(k-1)*(el2-1)*n2+(j-1)*el2*n2+(i-1)*(el2-1)*(n1-1)*n2-(j-1)*n2*(el2-1)+(i-1)*n2*(n1-1)] += val[1+(k-1)*n2:k*n2]
            end

            a2 += Δx2 
        end

        a1 += Δx1
    end
    return uh
end

function L2_prod_d(f::Function, basis::MultipatchBSplineTensorProductBasis{2, dirichlet})
    a1 = basis[1].interval.left
    b1 = basis[1].interval.right
    el1 = basis[1].el
    Δx1 = (b1-a1)/basis[1].el
    n1 = length(basis[1].B.B)

    a2 = basis[2].interval.left
    b2 = basis[2].interval.right
    el2 = basis[2].el
    Δx2 = (b2-a2)/el2
    n2 = length(basis[2].B.B)

    uh = zeros(n1*el1*n2*el2)

    for i in 1:el1
        for j in 1:el2
            g = (x, y) -> f(a1 + Δx1 * x, a2 + Δx2*y)
            val = 1/(Δx1 * Δx2)^2 * L2_prod(g, basis[1].B ⊗ basis[2].B) 

            for k in 1:n1
                uh[1+(k-1)*n2+(k-1)*(el2-1)*n2+(j-1)*el2*n2+(i-1)*(el2-1)*n2*n1+(i-1)*n2*n1-(j-1)*n2*(el2-1):k*n2+(k-1)*(el2-1)*n2+(j-1)*el2*n2+(i-1)*(el2-1)*n1*n2-(j-1)*n2*(el2-1)+(i-1)*n2*n1] += val[1+(k-1)*n2:k*n2]
            end

            a2 += Δx2 
        end

        a1 += Δx1
    end

    return uh
end

function col_mat(basis::MultipatchBSplineTensorProductBasis{2}, mm::Int) 
    X,Y = basis.B

    x, M1 = col_mat(X, mm)
    y, M2 = col_mat(Y, mm)

    xy = [(a,b) for a in x for b in y]

    return xy, M1 ⊗ M2
end

function col_mat_mixed_x(basis::MultipatchBSplineTensorProductBasis{2}, mm::Int) 
    X,Y = basis.B

    x, M1 = col_mat_d(X, mm)
    y, M2 = col_mat(Y, mm)

    xy = [(a,b) for a in x for b in y]

    return xy, M1 ⊗ M2
end

function col_mat_mixed_y(basis::MultipatchBSplineTensorProductBasis{2}, mm::Int) 
    X,Y = basis.B

    x, M1 = col_mat(X, mm)
    y, M2 = col_mat_d(Y, mm)

    xy = [(a,b) for a in x for b in y]

    return xy, M1 ⊗ M2
end
function col_mat_d(basis::MultipatchBSplineTensorProductBasis{2}, mm::Int) 
    X,Y = basis.B

    x, M1 = col_mat_d(X, mm)
    y, M2 = col_mat_d(Y, mm)

    xy = [(a,b) for a in x for b in y]

    return xy, M1 ⊗ M2
end

#n-D Calculations
function mass_matrix(basis::MultipatchBSplineTensorProductBasis{d, T}) where {d, T}
    matrices = []
    for i in 1:d
        push!(matrices, mass_matrix(basis[i]))
    end

    return ⊗(matrices...)
end

function mass_matrix_d(basis::MultipatchBSplineTensorProductBasis{d, T}) where {d, T}
    matrices = []
    for i in 1:d
        push!(matrices, mass_matrix_d(basis[i]))
    end

    return ⊗(matrices...)
end

function stiffness_matrix(basis::MultipatchBSplineTensorProductBasis{d, T}) where {d, T}
    matrices = []
    for i in 1:d
        push!(matrices,  stiffness_matrix(basis[i]))
    end

    return ⊗(matrices...)
end

function disc_diff(T, basis::MultipatchBSplineTensorProductBasis{2, dirichlet}) 
    if T == grad
        N1 = (length(basis[1].B.B)-1)*basis[1].el+1
        N2 = (length(basis[2].B.B)-1)*basis[2].el+1

        D1 = der_mat(basis[1]) ⊗ circ_id(N2)
        D2 = circ_id(N1) ⊗ der_mat(basis[2])
    elseif T == curl
        N1 = length(basis[1]'.B.B)*basis[1].el
        N2 = length(basis[2]'.B.B)*basis[2].el

        D1 = der_mat(basis[1]) ⊗ circ_id(N2)
        D2 = circ_id(N1) ⊗ der_mat(basis[2])
    end

    return discrete_differential{T, 2}(SVector(D1, D2))
end

function disc_diff_d(T, basis::MultipatchBSplineTensorProductBasis{2, ty}) where ty
    if T == grad
        N1 = length(basis[1].B.B)*basis[1].el
        N2 = length(basis[2].B.B)*basis[2].el

        D1 = der_mat_d(basis[1]) ⊗ circ_id(N2)
        D2 = circ_id(N1) ⊗ der_mat_d(basis[2])
    elseif T == curl
        N1 = length(basis[1]'.B.B)*basis[1].el
        N2 = length(basis[2]'.B.B)*basis[2].el

        D1 = der_mat_d(basis[1]) ⊗ circ_id(N2)
        D2 = circ_id(N1) ⊗ der_mat_d(basis[2])
    end

    return discrete_differential{T, 2}(SVector(D1, D2))
end

function L2_proj(f::Function, basis::MultipatchBSplineTensorProductBasis{N, T}) where {N, T}
    M = mass_matrix(basis)
    fh = M\L2_prod(f, basis)
    return SVector{length(fh)}(fh)
end

function L2_proj_d(f::Function, basis::MultipatchBSplineTensorProductBasis{N, T}) where {N, T}
    M = mass_matrix_d(basis)
    fh = M\L2_prod_d(f, basis)
    return SVector{length(fh)}(fh)
end

function L2_proj(f::Function, basis::SArray{Tuple{2}, MultipatchBSplineTensorProductBasis{2, T}}) where {T}
    fh = []
    M = mass_matrix.(basis)

    for i in 1:2
        push!(fh, M[i]\L2_prod((x,y) -> f(x,y)[i], basis[i]))
    end

    return SVector(fh...)
end

function L2_proj_d(f::Function, basis::SArray{Tuple{2}, MultipatchBSplineTensorProductBasis{2, T}}) where {T}
    fh = []
    M = mass_matrix_d.(basis)

    for i in 1:2
        push!(fh, M[i]\L2_prod_d((x,y) -> f(x,y)[i], basis[i]))
    end

    return SVector(fh...)
end

function col_mat_proj(basis::MultipatchBSplineTensorProductBasis{2, T}) where {T}
    X,Y = basis.B
    N = (length(X.B.B), length(Y.B.B))

    x, M1 = col_mat(X, N[1])
    y, M2 = col_mat(Y, N[2])

    xy = [(a,b) for a in x for b in y]

    return xy, M1 ⊗ M2
end

function col_mat_proj_mixed_x(basis::MultipatchBSplineTensorProductBasis{2, T}) where {T}
    X,Y = basis.B
    N = (length(X.B.B), length(Y.B.B))

    x, M1 = col_mat_d(X, N[1])
    y, M2 = col_mat(Y, N[2])

    xy = [(a,b) for a in x for b in y]

    return xy, M1 ⊗ M2
end

function col_mat_proj_mixed_y(basis::MultipatchBSplineTensorProductBasis{2, T}) where {T}
    X,Y = basis.B
    N = (length(X.B.B), length(Y.B.B))

    x, M1 = col_mat(X, N[1])
    y, M2 = col_mat_d(Y, N[2])

    xy = [(a,b) for a in x for b in y]

    return xy, M1 ⊗ M2
end

function col_mat_proj_d(basis::MultipatchBSplineTensorProductBasis{2, T}) where {T}
    X,Y = basis.B
    N = (length(X.B.B), length(Y.B.B))

    x, M1 = col_mat_d(X, N[1])
    y, M2 = col_mat_d(Y, N[2])

    xy = [(a,b) for a in x for b in y]

    return xy, M1 ⊗ M2
end


function conforming_averaging_projection_matrix(basis::MultipatchBSplineTensorProductBasis{2, T}) where {T}
    X,Y = basis.B
    Mx = conforming_averaging_projection_matrix(X)
    My = conforming_averaging_projection_matrix(Y)

    return Mx ⊗ My
end

function conforming_averaging_projection_matrix_x(basis::MultipatchBSplineTensorProductBasis{2, T}) where {T}
    X,Y = basis.B

    Mx = conforming_averaging_projection_matrix(X)
    My = circ_id(Y.el*(length(Y.B.B)-1))

    return Mx ⊗ My
end

function conforming_averaging_projection_matrix_y(basis::MultipatchBSplineTensorProductBasis{2, T}) where {T}
    X,Y = basis.B

    Mx = circ_id(X.el*(length(X.B.B)-1))
    My = conforming_averaging_projection_matrix(Y)

    return Mx ⊗ My
end

#todo weighted mass matrices

function L2_prod(f::Function, basis::MultipatchBSplineTensorProductBasis{2, periodic})
    a1 = basis[1].interval.left
    b1 = basis[1].interval.right
    el1 = basis[1].el
    Δx1 = (b1-a1)/basis[1].el
    n1 = length(basis[1].B.B)

    a2 = basis[2].interval.left
    b2 = basis[2].interval.right
    el2 = basis[2].el
    Δx2 = (b2-a2)/el2
    n2 = length(basis[2].B.B)
    
    
    uh = zeros(((n1-1)*el1+1)*((n2-1)*el2+1))
    
    for i in 1:el1
        for j in 1:el2
            g = (x, y) -> f(a1 + Δx1 * x, a2 + Δx2*y)
            val = 1/(Δx1 * Δx2) * L2_prod(g, basis[1].B ⊗ basis[2].B) 

            for k in 1:n1
                uh[1+(k-1)*n2+(k-1)*(el2-1)*(n2-1)+(j-1)*el2*(n2-1)+(i-1)*(el2-1)*(n2-1)*(n1-1)+(i-1)*n2*(n1-1)-(j-1)*(n2-1)*(el2-1):k*n2+(k-1)*(el2-1)*(n2-1)+(j-1)*el2*(n2-1)+(i-1)*(el2-1)*(n1-1)*(n2-1)-(j-1)*(n2-1)*(el2-1)+(i-1)*n2*(n1-1)] += val[1+(k-1)*n2:k*n2]
            end

            a2 += Δx2 
        end

        a1 += Δx1
    end

    uh_new = zeros(((n1-1)*el1)*((n2-1)*el2))
    for i in 1:(n1-1)*el1
            uh_new[1+(n2-1)*el2*(i-1):i*el2*(n2-1)] += uh[1+(n2-1)*el2*(i-1)+(i-1):i*el2*(n2-1)+(i-1)]
            uh_new[1+(n2-1)*el2*(i-1)] += uh[1+(n2-1)*el2*i+(i-1)]
    end

    uh_new[1:el2*(n2-1)] += uh[end-el2*(n2-1):end-1]
    uh_new[1] += uh[end]

    return uh_new
end
 
function L2_prod_mixed_x(f::Function, basis::MultipatchBSplineTensorProductBasis{2, periodic})
    a1 = basis[1].interval.left
    b1 = basis[1].interval.right
    el1 = basis[1].el
    Δx1 = (b1-a1)/basis[1].el
    n1 = length(basis[1].B.B)

    a2 = basis[2].interval.left
    b2 = basis[2].interval.right
    el2 = basis[2].el
    Δx2 = (b2-a2)/el2
    n2 = length(basis[2].B.B)

    uh = zeros((n1*el1)*((n2-1)*el2+1))

    for i in 1:el1
        for j in 1:el2
            g = (x, y) -> f(a1 + Δx1 * x, a2 + Δx2*y)
            val = 1/(Δx1^2 * Δx2) * L2_prod(g, basis[1].B ⊗ basis[2].B) 

            for k in 1:n1
                uh[1+(k-1)*n2+(k-1)*(el2-1)*(n2-1)+(j-1)*el2*(n2-1)+(i-1)*(el2-1)*(n2-1)*n1+(i-1)*n2*n1-(j-1)*(n2-1)*(el2-1):k*n2+(k-1)*(el2-1)*(n2-1)+(j-1)*el2*(n2-1)+(i-1)*(el2-1)*n1*(n2-1)-(j-1)*(n2-1)*(el2-1)+(i-1)*n2*n1] += val[1+(k-1)*n2:k*n2]
            end

            a2 += Δx2 
        end

        a1 += Δx1
    end

    uh_new = zeros((n1*el1)*((n2-1)*el2))
    for i in 1:n1*el1
            uh_new[1+(n2-1)*el2*(i-1):i*el2*(n2-1)] += uh[1+(n2-1)*el2*(i-1)+(i-1):i*el2*(n2-1)+(i-1)]
            uh_new[1+(n2-1)*el2*(i-1)] += uh[1+(n2-1)*el2*i+(i-1)]
    end

    return uh_new

end

function L2_prod_mixed_y(f::Function, basis::MultipatchBSplineTensorProductBasis{2, periodic})
    a1 = basis[1].interval.left
    b1 = basis[1].interval.right
    el1 = basis[1].el
    Δx1 = (b1-a1)/basis[1].el
    n1 = length(basis[1].B.B)

    a2 = basis[2].interval.left
    b2 = basis[2].interval.right
    el2 = basis[2].el
    Δx2 = (b2-a2)/el2
    n2 = length(basis[2].B.B)

    uh = zeros(((n1-1)*el1+1)*n2*el2)

    for i in 1:el1
        for j in 1:el2
            g = (x, y) -> f(a1 + Δx1 * x, a2 + Δx2*y)
            val = 1/(Δx1 * Δx2^2) * L2_prod(g, basis[1].B ⊗ basis[2].B) 

            for k in 1:n1
                uh[1+(k-1)*n2+(k-1)*(el2-1)*n2+(j-1)*el2*n2+(i-1)*(el2-1)*n2*(n1-1)+(i-1)*n2*(n1-1)-(j-1)*n2*(el2-1):k*n2+(k-1)*(el2-1)*n2+(j-1)*el2*n2+(i-1)*(el2-1)*(n1-1)*n2-(j-1)*n2*(el2-1)+(i-1)*n2*(n1-1)] += val[1+(k-1)*n2:k*n2]
            end

            a2 += Δx2 
        end

        a1 += Δx1
    end

    uh_new = zeros(((n1-1)*el1)*n2*el2)
    for i in 1:(n1-1)*el1
            uh_new[1+(n2)*el2*(i-1):i*el2*(n2)] += uh[1+(n2)*el2*(i-1):i*el2*(n2)]
    end

    uh_new[1:(n2)*el2] += uh[end-(n2)*el2+1:end]

    return uh_new
end

function L2_prod_d(f::Function, basis::MultipatchBSplineTensorProductBasis{2, periodic})
    a1 = basis[1].interval.left
    b1 = basis[1].interval.right
    el1 = basis[1].el
    Δx1 = (b1-a1)/basis[1].el
    n1 = length(basis[1].B.B)

    a2 = basis[2].interval.left
    b2 = basis[2].interval.right
    el2 = basis[2].el
    Δx2 = (b2-a2)/el2
    n2 = length(basis[2].B.B)

    uh = zeros(n1*el1*n2*el2)

    for i in 1:el1
        for j in 1:el2
            g = (x, y) -> f(a1 + Δx1 * x, a2 + Δx2*y)
            val = 1/(Δx1 * Δx2)^2 * L2_prod(g, basis[1].B ⊗ basis[2].B) 

            for k in 1:n1
                uh[1+(k-1)*n2+(k-1)*(el2-1)*n2+(j-1)*el2*n2+(i-1)*(el2-1)*n2*n1+(i-1)*n2*n1-(j-1)*n2*(el2-1):k*n2+(k-1)*(el2-1)*n2+(j-1)*el2*n2+(i-1)*(el2-1)*n1*n2-(j-1)*n2*(el2-1)+(i-1)*n2*n1] += val[1+(k-1)*n2:k*n2]
            end

            a2 += Δx2 
        end

        a1 += Δx1
    end

    return uh
end

function disc_diff(T, basis::MultipatchBSplineTensorProductBasis{2, periodic}) 
    if T == grad
        N1 = (length(basis[1].B.B)-1)*basis[1].el
        N2 = (length(basis[2].B.B)-1)*basis[2].el

        D1 = der_mat(basis[1]) ⊗ circ_id(N2)
        D2 = circ_id(N1) ⊗ der_mat(basis[2])
    elseif T == curl
        N1 = length(basis[1]'.B.B)*basis[1].el
        N2 = length(basis[2]'.B.B)*basis[2].el

        D1 = der_mat(basis[1]) ⊗ circ_id(N2)
        D2 = circ_id(N1) ⊗ der_mat(basis[2])
    end

    return discrete_differential{T, 2}(SVector(D1, D2))
end

#(X', Y) and (X, Y')
function mixed_mass_matrix_x(basis1::MultipatchBSplineTensorProductBasis{2, periodic}, basis2::MultipatchBSplineTensorProductBasis{2, periodic}) #where T
    X1, Y1 = basis1.B 
    X2, Y2 = basis2.B

    # x basis functions
    a1 = X1.interval.left
    b1 = X1.interval.right
    n1 = length(X1.B.B)
    el1 = X1.el
    Δx1 = (b1-a1)/el1
    

    a2 = X2.interval.left
    b2 = X2.interval.right
    n2 = length(X2.B.B)
    el2 = X2.el
    Δx2 = (b2-a2)/el2
    
    Mx = mass_matrix(X1.B, X2.B)
    MMx = zeros(n1*el1, (n2-1)*el2+1)

    for i in 1:el1
        for j in 1:el2
            MMx[1+(i-1)*n1:i*n1,1+(j-1)*(n2-1):j*(n2-1)+1] += Mx
        end
    end

    #periodic in x in 2nd basis function
    M_px = zero(MMx[1:end, 1:end-1])
    M_px += MMx[1:end, 1:end-1]
    M_px[1:end, 1] += MMx[:, end]


    # y basis functions
    a1 = Y1.interval.left
    b1 = Y1.interval.right
    n1 = length(Y1.B.B)
    el1 = Y1.el
    Δy1 = (b1-a1)/el1
    

    a2 = Y2.interval.left
    b2 = Y2.interval.right
    n2 = length(Y2.B.B)
    el2 = Y2.el
    Δy2 = (b2-a2)/el2
    
    My = mass_matrix(Y1.B, Y2.B)
    MMy = zeros((n1-1)*el1+1, n2*el2)

    for i in 1:el1
        for j in 1:el2
            MMy[1+(i-1)*(n1-1):i*(n1-1)+1,1+(j-1)*n2:j*n2] += My
        end
    end

    #periodic in x in 2nd basis function
    M_py = zero(MMy[1:end-1, 1:end])
    M_py += MMy[1:end-1, 1:end]
    M_py[1, 1:end] += MMy[end, :]

    return 1/Δx1 * 1/Δx2 * M_px ⊗ M_py
end

#(X', Y) and (X, Y')
function mixed_mass_matrix_x_d(basis1::MultipatchBSplineTensorProductBasis{2, periodic}, basis2::MultipatchBSplineTensorProductBasis{2, periodic}) #where T
    X1, Y1 = basis1.B 
    X2, Y2 = basis2.B

    # x basis functions
    a1 = X1.interval.left
    b1 = X1.interval.right
    n1 = length(X1.B.B)
    el1 = X1.el
    Δx1 = (b1-a1)/el1
    

    a2 = X2.interval.left
    b2 = X2.interval.right
    n2 = length(X2.B.B)
    el2 = X2.el
    Δx2 = (b2-a2)/el2
    
    Mx = mass_matrix(X1.B, X2.B)
    MMx = zeros(n1*el1, n2*el2)

    for i in 1:el1
        for j in 1:el2
            MMx[1+(i-1)*n1:i*n1,1+(j-1)*n2:j*n2] += Mx
        end
    end

    # y basis functions
    a1 = Y1.interval.left
    b1 = Y1.interval.right
    n1 = length(Y1.B.B)
    el1 = Y1.el
    Δy1 = (b1-a1)/el1
    

    a2 = Y2.interval.left
    b2 = Y2.interval.right
    n2 = length(Y2.B.B)
    el2 = Y2.el
    Δy2 = (b2-a2)/el2
    
    My = mass_matrix(Y1.B, Y2.B)
    MMy = zeros(n1*el1, n2*el2)

    for i in 1:el1
        for j in 1:el2
            MMy[1+(i-1)*n1:i*n1,1+(j-1)*n2:j*n2] += My
        end
    end


    return 1/Δx1^2 * 1/Δx2^2 * MMx ⊗ MMy
end

function L2_proj_mixed_V1(f::Function, basis::SArray{Tuple{2}, MultipatchBSplineTensorProductBasis{2, T}}) where {T}
    X, Y = basis[1].B
    M1x = mass_matrix_d(X)
    M1y = mass_matrix(Y)
    M1_1 = M1x ⊗ M1y

    fh_1 = M1_1\L2_prod_mixed_x((x,y) -> f(x,y)[1], basis[1])

    X, Y = basis[2].B
    M1x = mass_matrix(X)
    M1y = mass_matrix_d(Y)
    M1_2 = M1x ⊗ M1y

    fh_2 = M1_2\L2_prod_mixed_y((x,y) -> f(x,y)[2], basis[2])

    return SVector(fh_1, fh_2)
end


