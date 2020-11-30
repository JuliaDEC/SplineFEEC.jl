# f' = df
function test_derivative(k::Int, N::Int)
    f(x) = sin(2π * x)
    df(x) = 2π * cos(2π * x)

    interval = 0..1
    basis = PeriodicBSpline(interval, k, N)

    M = basis ⊙ basis
    M_d = mass_matrix(basis')
    D = der_mat(N)

    fh = M \ (f ⊙ basis)
    dfh = M_d \ (df ⊙ basis')

    # Shouldn't be the following numerically exact? -> care for integral f ⊙ basis 
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

function Maxwells_2d()

    domain = (0..1) × (0..1) 
    k = (4, 5)
    N = (20, 15)
    E(t, x, y) = sin(2π * t) .* SVector(sin(2π * y), sin(2π * x))
    B(t, x, y) = cos(2π * t) * (-cos(x*2π) + cos(2π * y))

    X,Y = PeriodicBSplineTensorProductBasis(domain, k, N).B
    V0 = X ⊗ Y
    V1 = SVector(X' ⊗ Y, X ⊗ Y')
    V2 = X' ⊗ Y'

    M1 = mass_matrix.(V1) #Vector representing the diagonal of the mass blockmatrix
    M2 = mass_matrix(V2)
    M1inv = inv.(M1) #using the circulant inverse
    C = disc_diff(curl, N)
        
    Eh = L2_proj((x,y) -> E(0, x, y), V1)
    Bh = L2_proj((x,y) -> B(0, x, y), V2)

    #prepare time stepping
    it = 123
    Δt = 0.01
    T = Δt * it

    #simple leap frog scheme
    for i in 1:it
        Eh -= Δt/2 * M1inv .* (C' * SVector{length(Bh)}(M2*Bh)) #need to rework the typing with svectors and kronecker matrices
        Bh += Δt * (C * Eh)
        Eh -= Δt/2 * M1inv .* (C' * SVector{length(Bh)}(M2*Bh))
    end

    #calculate the error in the different components
    (xy1_1, colmat1_1), (xy1_2, colmat1_2) = col_mat.(V1, 200)
    xy2, colmat2 = col_mat(V2, 200)
    val_E1 = (i -> E.(T, xy1_1[i][1], xy1_1[i][2])[1]).(1:length(xy1_1))
    val_E2 = (i -> E.(T, xy1_2[i][1], xy1_2[i][2])[2]).(1:length(xy1_2))
    val_B = (i -> B.(T, xy2[i][1], xy2[i][2])).(1:length(xy2))
    @show maximum(abs.(colmat2 * Bh - val_B))
    @show maximum(abs.(colmat1_1 * Eh[1] - val_E1))
    @show maximum(abs.(colmat1_2 * Eh[2] - val_E2))

    return Eh, Bh
end

function Maxwells_3d()
    
    domain = (0..1) × (0..1) × (0..1) 
    k = (4, 5, 3)
    N = (7, 8, 6)
    E(t, x, y, z) = sin(2π * t) .* SVector(sin(2π * y), sin(2π * x), 0)
    B(t, x, y, z) = cos(2π * t) .* SVector(0, 0, -cos(x*2π) + cos(2π * y))

    X,Y, Z = PeriodicBSplineTensorProductBasis(domain, k, N).B
    V0 = X ⊗ Y ⊗ Z
    V1 = SVector(X' ⊗ Y ⊗ Z, X ⊗ Y' ⊗ Z, X ⊗ Y ⊗ Z')
    V2 = SVector(X ⊗ Y' ⊗ Z', X' ⊗ Y ⊗ Z', X' ⊗ Y' ⊗ Z)
    V3 = X' ⊗ Y' ⊗ Z'

    M1 = mass_matrix.(V1) #Vector representing the diagonal of the mass blockmatrix
    M2 = mass_matrix.(V2)
    M1inv = inv.(M1) #using the circulant inverse
    C = disc_diff(curl, N)
        
    Eh = L2_proj((x,y, z) -> E(0, x, y, z), V1)
    Bh = L2_proj((x,y, z) -> B(0, x, y, z), V2)

    #prepare time stepping
    it = 20
    Δt = 0.01
    T = Δt * it

    #simple leap frog scheme
    for i in 1:it
        Eh += Δt/2 * M1inv .* (C' * (M2 .* Bh)) #need to rework the typing with svectors and kronecker matrices
        Bh -= Δt * (C * Eh)
        Eh += Δt/2 * M1inv .* (C' * (M2 .* Bh))
    end

    #calculate the error in the different components
    (xyz1_1, colmat1_1), (xyz1_2, colmat1_2), (xyz1_3, colmat1_3) = col_mat.(V1, 100)
    (xyz2_1, colmat2_1), (xyz2_2, colmat2_2), (xyz2_3, colmat2_3) = col_mat.(V2, 100)
    val_E1 = (i -> E.(T, xyz1_1[i][1], xyz1_1[i][2], xyz1_1[i][3])[1]).(1:length(xyz1_1))
    val_E2 = (i -> E.(T, xyz1_2[i][1], xyz1_2[i][2], xyz1_2[i][3])[2]).(1:length(xyz1_2))
    val_E3 = (i -> E.(T, xyz1_3[i][1], xyz1_3[i][2], xyz1_3[i][3])[3]).(1:length(xyz1_3))
    val_B1 = (i -> B.(T, xyz2_1[i][1], xyz2_1[i][2], xyz2_1[i][3])[1]).(1:length(xyz2_1))
    val_B2 = (i -> B.(T, xyz2_2[i][1], xyz2_2[i][2], xyz2_2[i][3])[2]).(1:length(xyz2_2))
    val_B3 = (i -> B.(T, xyz2_3[i][1], xyz2_3[i][2], xyz2_3[i][3])[3]).(1:length(xyz2_3))
    @show maximum(abs.(colmat1_1 * Eh[1] - val_E1))
    @show maximum(abs.(colmat1_2 * Eh[2] - val_E2))
    @show maximum(abs.(colmat1_3 * Eh[3] - val_E3))
    @show maximum(abs.(colmat2_1 * Bh[1] - val_B1))
    @show maximum(abs.(colmat2_2 * Bh[2] - val_B2))
    @show maximum(abs.(colmat2_3 * Bh[3] - val_B3))
    return Eh, Bh
end