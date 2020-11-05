# f' = df
function test_derivative(k::Int, N::Int)
    f(x) = sin(2π * x)
    df(x) = 2π * cos(2π * x)

    interval = 0..1
    basis= PeriodicBSpline(interval, k, N)

    M = basis ⊙ basis
    M_d = mass_matrix(basis')
    D = der_mat(N)

    fh = M \ (f ⊙ basis)
    dfh = M_d \ (df ⊙ basis')

    # Shouldn't be the following numerically exact? 
    @show D * fh ≈ dfh
    @show maximum(abs.( D * fh - dfh))
    
    x, colmat = col_mat(basis, 100)
    x, colmat_d = col_mat(basis', 100) 

    @show maximum(abs.(colmat * fh - f.(x)))
    @show maximum(abs.(colmat_d * dfh - df.(x)))
    @show maximum(abs.(colmat_d * D * fh - df.(x)))
end

# -Δu = f
function Poisson_1d(k::Int, N::Int)
    u(x) = sin(2π * x)
    f(x) = -4π^2 * sin(2π * x)

    interval = 0..1
    basis = PeriodicBSpline(interval, k, N)

    Δ = ∇(basis) ⊙ ∇(basis) # Equivalent to: stiffness_matrix(basis')
    R = 1/N .* circ_ones(N)
    P = circ_id(N) - R
    fh = f ⊙ basis
    uh = -(Δ + R) \ (P * fh)

    x, colmat = col_mat(basis, 200)

    @show maximum( abs.(colmat * uh .- u.(x) )) 
    return x, colmat*uh
end

# Example usage: L2_proj_2d((x,y) -> sin(2π*x)*cos(2π*y), (-1..1) × (0..1), (4,3), (30,20))
function L2_proj_2d(f::Function, domain::ProductDomain{SVector{2,T}}, k::NTuple{2,Int}, N::NTuple{2,Int}) where {T}
    # Alternatively: 
    #domain = domain.domains
    #X = PeriodicBSpline(domain[1], k[1], N[1])
    #Y = PeriodicBSpline(domain[2], k[2], N[2])
    #basis = X ⊗ Y 
    basis = PeriodicBSplineTensorProductBasis(domain, k, N)

    M = basis ⊙ basis # Equivalent to: mass_matrix(basis)
    fh = M \ (f ⊙ basis)
    
    xy, colmat = col_mat(basis, 100)
    val_f = (i -> f.(xy[i][1], xy[i][2])).(1:length(xy))
    @show maximum(abs.(val_f - colmat * fh))

    #example for plotting: plot3d( (i -> xy[i][1]).(1:length(xy)), (i -> xy[i][2]).(1:1:length(xy)), colmat * fh); surface!(x, y, f)
    return xy, colmat * fh
end

# -Δu = f
function Poisson_2d(k::NTuple{2,Int}, N::NTuple{2,Int})
    u(x, y) = sin(2π * x) + sin(2π * y)
    f(x, y) = -4π^2 * sin(2π * x) - 4π^2 * sin(2π * y)
    domain = (0..1) × (-1..1)

    basis = PeriodicBSplineTensorProductBasis(domain, k, N)

    Δ = ∇(basis) ⊙ ∇(basis)
    
    NN = prod(N)
    R = 1/NN .* circ_ones(NN)
    P = circ_id(NN) - R

    fh = f ⊙ basis

    uh = -(Δ + R) \ (P * fh)

    xy, colmat = col_mat(basis, 100)

    val_u = (i -> u.(xy[i][1], xy[i][2])).(1:length(xy))
    @show maximum(abs.(val_u - colmat * uh))

    return xy, colmat * uh
end

function L2_proj_3d(f::Function, domain::ProductDomain{SVector{3,T}}, k::NTuple{3,Int}, N::NTuple{3,Int}) where {T}
    basis = PeriodicBSplineTensorProductBasis(domain, k, N)

    @time M = basis ⊙ basis 
    @time fh_ = f ⊙ basis
    @time fh = M \ fh_

    @time xyz, colmat = col_mat(basis, 100)
    val_f = (i -> f.(xyz[i][1], xyz[i][2], xyz[i][3])).(1:length(xyz))
    @show maximum(abs.(val_f - colmat * fh))

    return xyz, colmat * fh
end