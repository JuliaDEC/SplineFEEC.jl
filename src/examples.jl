function test_derivatives_projections_1d(k::Int, N::Int, el::Int)
    f(x) = sin(2π * x)
    df(x) = 2π * cos(2π * x)
    interval = 0..1

    print("Periodic B-Spline Basis:\n")
    basis = PeriodicBSpline(interval, k, N)

    M = basis ⊙ basis
    M_d = mass_matrix(basis')
    D = der_mat(basis)

    fh = M \ (f ⊙ basis)
    dfh = M_d \ (df ⊙ basis')

    @show maximum(abs.( D * fh - dfh))
    
    x, colmat = col_mat(basis, 100)
    x, colmat_d = col_mat(basis', 100) 

    @show maximum(abs.(colmat * fh - f.(x)))
    @show maximum(abs.(colmat_d * dfh - df.(x)))
    @show maximum(abs.(colmat_d * D * fh - df.(x)))

    S = stiffness_matrix(basis')
    @show maximum(abs.(D' * M_d * D - S))

    print("Dirichlet B-Spline Basis:\n")
    basis = DirichletBSpline(interval, k, N)

    M = mass_matrix(basis)
    M_d = mass_matrix(basis')
    D = SplineFEEC.der_mat(basis)
    
    fh = M \ (f ⊙ basis)
    dfh = M_d \ (df ⊙ basis')
    
    @show maximum(abs.( D * fh - dfh))
    
    x, colmat = col_mat(basis, 100)
    x, colmat_d = col_mat(basis', 100) 

    @show maximum(abs.(colmat * fh - f.(x)))
    @show maximum(abs.(colmat_d * dfh - df.(x)))
    @show maximum(abs.(colmat_d * D * fh - df.(x)))
        
    S = stiffness_matrix(basis')
    @show maximum(abs.(D' * M_d * D - S))

    print("Multi-patch Dirichlet B-Spline Basis: \n")

    Δx = (interval.right - interval.left)/el
    basis = DirichletMultipatchBSpline(interval, k, N, el)

    M = mass_matrix(basis)
    M_d =  mass_matrix_d(basis')
    D = der_mat(basis)
    fh = M \ L2_prod(f, basis)
    dfh = M_d\ L2_prod_d(df, basis')

    @show maximum(abs.( D * fh - dfh))
    
    x, colmat = col_mat(basis, 100)
    x_d, colmat_d = col_mat_d(basis', 100) 

    @show maximum(abs.(colmat * fh - f.(x)))
    @show maximum(abs.(colmat_d * dfh - df.(x_d)))
    @show maximum(abs.(colmat_d * D * fh - df.(x_d)))
    S = stiffness_matrix(basis')

    @show maximum(abs.( D' * M_d * D - S))

    print("Multi-patch Periodic B-Spline Basis: \n")

    Δx = (interval.right - interval.left)/el
    basis = PeriodicMultipatchBSpline(interval, k, N, el)

    M = mass_matrix(basis)
    M_d =  mass_matrix_d(basis')
    D = der_mat(basis)
    fh = M \ L2_prod(f, basis)
    dfh = M_d\ L2_prod_d(df, basis')

    @show maximum(abs.( D * fh - dfh))
    
    x, colmat = col_mat(basis, 100)
    x_d, colmat_d = col_mat_d(basis', 100) 

    @show maximum(abs.(colmat * fh - f.(x)))
    @show maximum(abs.(colmat_d * dfh - df.(x_d)))
    @show maximum(abs.(colmat_d * D * fh - df.(x_d)))
    S = stiffness_matrix(basis')

    @show maximum(abs.( D' * M_d * D - S))
end

function test_derivatives_projections_2d(k::NTuple{2,Int}, N::NTuple{2,Int})
    f(x,y) = sin(2π * x)*sin(2π * y)
    gradf(x,y) = 2π * SVector(cos(2π * x)*sin(2π * y), 
                                sin(2π * x)*cos(2π * y))

    g(x,y) = SVector(sin(2π * x)*sin(2π * y),
                        sin(2π * y)*sin(2π * x))
    curlg(x,y) = 2π * (sin(2π * y)*cos(2π * x) -  sin(2π * x)*cos(2π * y))
    domain = (0..1) × (0..1)

    print("Periodic B-Spline Basis: \n")
    X, Y = PeriodicBSplineTensorProductBasis(domain, k, N).B
    V0 = X ⊗ Y 
    V1 = SVector(X' ⊗ Y , X ⊗ Y')
    V2 = X' ⊗ Y' 

    G = disc_grad(V0)
    C = disc_curl(V0)

    xy0, colmat0 = col_mat(V0, 100)
    (xy1_1, colmat1_1), (xy1_2, colmat1_2) = col_mat.(V1, 100) 
    xy2, colmat2 = col_mat(V2, 100)

    fh = L2_proj(f, V0)
    gradfh = L2_proj(gradf, V1)

    @show maximum(abs.(vcat((G * fh - gradfh)...)))

    val_f = (i -> f.(xy0[i][1], xy0[i][2])).(1:length(xy0))
    @show maximum(abs.(colmat0 * fh - val_f))

    val_gradf1 = (i -> gradf.(xy1_1[i][1], xy1_1[i][2])[1]).(1:length(xy1_1))
    val_gradf2 = (i -> gradf.(xy1_2[i][1], xy1_2[i][2])[2]).(1:length(xy1_2))
    #return fh, gradfh, colmat1_1, val_gradf1

    @show maximum(abs.(colmat1_1 * gradfh[1] - val_gradf1))
    @show maximum(abs.(colmat1_2 * gradfh[2] - val_gradf2))


    gh = L2_proj(g, V1)
    curlgh = L2_proj(curlg, V2)
    @show maximum(abs.((C * gh - curlgh)))

    val_g1 = (i -> g.(xy1_1[i][1], xy1_1[i][2])[1]).(1:length(xy1_1))
    val_g2 = (i -> g.(xy1_2[i][1], xy1_2[i][2])[2]).(1:length(xy1_2))
    @show maximum(abs.(colmat1_1 * gh[1] - val_g1))
    @show maximum(abs.(colmat1_2 * gh[2] - val_g2))
      
    val_curlg = (i -> curlg.(xy2[i][1], xy2[i][2])).(1:length(xy2))
    @show maximum(abs.(colmat2 * curlgh - val_curlg))

    print("Dirichlet B-Spline Basis: \n")
    X, Y = DirichletBSplineTensorProductBasis(domain, k, N).B
    V0 = X ⊗ Y 
    V1 = SVector(X' ⊗ Y , X ⊗ Y')
    V2 = X' ⊗ Y' 

    G = disc_grad(V0)
    C = disc_curl(V0)

    xy0, colmat0 = col_mat(V0, 100)
    (xy1_1, colmat1_1), (xy1_2, colmat1_2) = col_mat.(V1, 100) 
    xy2, colmat2 = col_mat(V2, 100)

    fh = L2_proj(f, V0)
    gradfh = L2_proj(gradf, V1)

    @show maximum(abs.(vcat((G * fh - gradfh)...)))

    val_f = (i -> f.(xy0[i][1], xy0[i][2])).(1:length(xy0))
    @show maximum(abs.(colmat0 * fh - val_f))

    val_gradf1 = (i -> gradf.(xy1_1[i][1], xy1_1[i][2])[1]).(1:length(xy1_1))
    val_gradf2 = (i -> gradf.(xy1_2[i][1], xy1_2[i][2])[2]).(1:length(xy1_2))

    @show maximum(abs.(colmat1_1 * gradfh[1] - val_gradf1))
    @show maximum(abs.(colmat1_2 * gradfh[2] - val_gradf2))

    gh = L2_proj(g, V1)
    curlgh = L2_proj(curlg, V2)
    @show maximum(abs.((C * gh - curlgh)))

    val_g1 = (i -> g.(xy1_1[i][1], xy1_1[i][2])[1]).(1:length(xy1_1))
    val_g2 = (i -> g.(xy1_2[i][1], xy1_2[i][2])[2]).(1:length(xy1_2))
    @show maximum(abs.(colmat1_1 * gh[1] - val_g1))
    @show maximum(abs.(colmat1_2 * gh[2] - val_g2))
      
    val_curlg = (i -> curlg.(xy2[i][1], xy2[i][2])).(1:length(xy2))
    @show maximum(abs.(colmat2 * curlgh - val_curlg))

    #test collocation projections
    xy0, colmat0 = col_mat_proj(V0)
    (xy1_1, colmat1_1), (xy1_2, colmat1_2) = col_mat_proj.(V1)
    xy2, colmat2 = col_mat_proj(V2)

    val_f = (i -> f.(xy0[i][1], xy0[i][2])).(1:length(xy0))
    @show maximum(abs.(colmat0\val_f - fh))

    val_gradf1 = (i -> gradf.(xy1_1[i][1], xy1_1[i][2])[1]).(1:length(xy1_1))
    val_gradf2 = (i -> gradf.(xy1_2[i][1], xy1_2[i][2])[2]).(1:length(xy1_2))
    @show maximum(abs.(colmat1_1\val_gradf1 - gradfh[1]))
    @show maximum(abs.(colmat1_2\val_gradf2 - gradfh[2]))

    val_g1 = (i -> g.(xy1_1[i][1], xy1_1[i][2])[1]).(1:length(xy1_1))
    val_g2 = (i -> g.(xy1_2[i][1], xy1_2[i][2])[2]).(1:length(xy1_2))
    @show maximum(abs.(colmat1_1\val_g1 - gh[1]))
    @show maximum(abs.(colmat1_2\val_g2 - gh[2]))

    val_curlg = (i -> curlg.(xy2[i][1], xy2[i][2])).(1:length(xy2))
    @show maximum(abs.(colmat2\val_curlg - curlgh))

end

function test_derivatives_projections_Multipatch_2d(k::NTuple{2,Int}, N::NTuple{2,Int}, el::NTuple{2,Int})
    f(x,y) = sin(2π * x)*sin(2π * y)
    gradf(x,y) = 2π * SVector(cos(2π * x)*sin(2π * y), 
                                sin(2π * x)*cos(2π * y))

    g(x,y) = SVector(sin(2π * x)*sin(2π * y),
                        sin(2π * y)*sin(2π * x))
    curlg(x,y) = 2π * (sin(2π * y)*cos(2π * x) -  sin(2π * x)*cos(2π * y))
    domain = (0..1) × (0..1)

    print("Multipatch Dirichlet B-Spline Basis: \n")
    X, Y = DirichletMultipatchTensorProductBSpline(domain, k, N, el).B
    V0 = X ⊗ Y 
    V1 = SVector(X' ⊗ Y , X ⊗ Y')
    V2 = X' ⊗ Y' 

    G = disc_diff(grad, V0)
    C = disc_diff(curl, V0)

    xy0, colmat0 = col_mat(V0, 100)

    #need a mixed setup
    x, cmx = col_mat_d(X', 100)
    y, cmy = col_mat(Y, 100)
    xy1_1 = [(a,b) for a in x for b in y]
    colmat1_1 = cmx ⊗ cmy 

    x, cmx = col_mat(X, 100)
    y, cmy = col_mat_d(Y', 100)
    xy1_2 = [(a,b) for a in x for b in y]
    colmat1_2 = cmx ⊗ cmy 

    xy2, colmat2 = col_mat_d(V2, 100)

    fh = L2_proj(f, V0)

    #also mixed setup TODO introduce better functions
    M1x = mass_matrix_d(X')
    M1y = mass_matrix(Y)
    M1_1 = M1x ⊗ M1y
    gradfh1 = M1_1\L2_prod_mixed_x((x,y) -> gradf(x,y)[1], X' ⊗ Y)

    M2x = mass_matrix(X)
    M2y = mass_matrix_d(Y')
    M1_2 = M2x ⊗ M2y
    gradfh2 = M1_2\L2_prod_mixed_y((x,y) -> gradf(x,y)[2], X ⊗ Y')
    gradfh = SVector(gradfh1, gradfh2)
    @show maximum(abs.(vcat((G * fh - gradfh)...)))

    val_f = (i -> f.(xy0[i][1], xy0[i][2])).(1:length(xy0))
    @show maximum(abs.(colmat0 * fh - val_f))

    val_gradf1 = (i -> gradf.(xy1_1[i][1], xy1_1[i][2])[1]).(1:length(xy1_1))
    val_gradf2 = (i -> gradf.(xy1_2[i][1], xy1_2[i][2])[2]).(1:length(xy1_2))

    @show maximum(abs.(colmat1_1 * gradfh[1] - val_gradf1))
    @show maximum(abs.(colmat1_2 * gradfh[2] - val_gradf2))

    #gh = L2_proj(g, V1)
    gh = SVector( M1_1\L2_prod_mixed_x((x,y) -> g(x,y)[1], X' ⊗ Y),
                  M1_2\L2_prod_mixed_y((x,y) -> g(x,y)[2], X ⊗ Y'))


    curlgh = L2_proj_d(curlg, V2)
    @show maximum(abs.((C * gh - curlgh)))

    val_g1 = (i -> g.(xy1_1[i][1], xy1_1[i][2])[1]).(1:length(xy1_1))
    val_g2 = (i -> g.(xy1_2[i][1], xy1_2[i][2])[2]).(1:length(xy1_2))
    @show maximum(abs.(colmat1_1 * gh[1] - val_g1))
    @show maximum(abs.(colmat1_2 * gh[2] - val_g2))
      
    val_curlg = (i -> curlg.(xy2[i][1], xy2[i][2])).(1:length(xy2))
    @show maximum(abs.(colmat2 * curlgh - val_curlg))

    #test collocation projections
    xy0, colmat0 = col_mat_proj(V0)
    xy1_1, colmat1_1 = col_mat_proj_mixed_x(V1[1])
    xy1_2, colmat1_2 = col_mat_proj_mixed_y(V1[2]) 
    xy2, colmat2 = col_mat_proj_d(V2)

    val_f = (i -> f.(xy0[i][1], xy0[i][2])).(1:length(xy0))
    @show maximum(abs.(colmat0\val_f - fh))

    val_gradf1 = (i -> gradf.(xy1_1[i][1], xy1_1[i][2])[1]).(1:length(xy1_1))
    val_gradf2 = (i -> gradf.(xy1_2[i][1], xy1_2[i][2])[2]).(1:length(xy1_2))
    @show maximum(abs.(colmat1_1\val_gradf1 - gradfh[1]))
    @show maximum(abs.(colmat1_2\val_gradf2 - gradfh[2]))

    val_g1 = (i -> g.(xy1_1[i][1], xy1_1[i][2])[1]).(1:length(xy1_1))
    val_g2 = (i -> g.(xy1_2[i][1], xy1_2[i][2])[2]).(1:length(xy1_2))
    @show maximum(abs.(colmat1_1\val_g1 - gh[1]))
    @show maximum(abs.(colmat1_2\val_g2 - gh[2]))

    val_curlg = (i -> curlg.(xy2[i][1], xy2[i][2])).(1:length(xy2))
    @show maximum(abs.(colmat2\val_curlg - curlgh))

    print("\nConga Dirichlet B-Spline Basis: \n")

    Δx = (domain.domains[1].right - domain.domains[1].left)/el[1]
    Δy = (domain.domains[2].right - domain.domains[2].left)/el[2]
   
    X, Y = DirichletMultipatchTensorProductBSpline(domain, k, N, el).B
    V0 = X ⊗ Y 
    V1 = SVector(X' ⊗ Y , X ⊗ Y')
    V2 = X' ⊗ Y' 

    G = disc_diff(grad, V0)
    C = disc_diff(curl, V0)
    P = conforming_averaging_projection_matrix(V0)
    Px = conforming_averaging_projection_matrix_x(V0)
    Py = conforming_averaging_projection_matrix_y(V0)

    xy0, colmat0 = col_mat(V0, 100)
    xy1_1, colmat1_1 = col_mat_mixed_x(V1[1], 100) 
    xy1_2, colmat1_2 = col_mat_mixed_y(V1[2], 100) 
    xy2, colmat2 = col_mat_d(V2, 100)

    fh = L2_proj_d(f, V0)
    fh *= 1/(Δx*Δy)
    
    gradfh = L2_proj_d(gradf, V1)
    gradfh = SVector(1/Δy *gradfh[1],1/Δx * gradfh[2])
    @show maximum(abs.(vcat(( G * SVector{size(P)[1]}(P * fh) - SVector(Py * gradfh[1],Px * gradfh[2]) )...)))

    val_f = (i -> f.(xy0[i][1], xy0[i][2])).(1:length(xy0))
    @show maximum(abs.(colmat0 * P * fh - val_f))

    val_gradf1 = (i -> gradf.(xy1_1[i][1], xy1_1[i][2])[1]).(1:length(xy1_1))
    val_gradf2 = (i -> gradf.(xy1_2[i][1], xy1_2[i][2])[2]).(1:length(xy1_2))
    @show maximum(abs.(colmat1_1 * Py * gradfh[1] - val_gradf1))
    @show maximum(abs.(colmat1_2 * Px * gradfh[2] - val_gradf2))

    gh = L2_proj_d(g, V1)
    gh = SVector(1/Δy * gh[1], 1/Δx * gh[2])
    curlgh = L2_proj_d(curlg, V2)
    @show maximum(abs.((C * SVector(Py * gh[1], Px * gh[2] ) - curlgh)))

    val_g1 = (i -> g.(xy1_1[i][1], xy1_1[i][2])[1]).(1:length(xy1_1))
    val_g2 = (i -> g.(xy1_2[i][1], xy1_2[i][2])[2]).(1:length(xy1_2))
    @show maximum(abs.(colmat1_1 * Py * gh[1] - val_g1))
    @show maximum(abs.(colmat1_2 * Px * gh[2] - val_g2))
      
    val_curlg = (i -> curlg.(xy2[i][1], xy2[i][2])).(1:length(xy2))
    @show maximum(abs.(colmat2 * curlgh - val_curlg))

    #testing the colocation projection
    xy0, colmat0 = col_mat_proj_d(V0)
    xy1_1, colmat1_1 = col_mat_proj_d(V1[1])
    xy1_2, colmat1_2 = col_mat_proj_d(V1[2]) 
    xy2, colmat2 = col_mat_proj_d(V2)

    val_f = (i -> f.(xy0[i][1], xy0[i][2])).(1:length(xy0))
    @show maximum(abs.((Δx*Δy) * colmat0\val_f - fh))

    val_gradf1 = (i -> gradf.(xy1_1[i][1], xy1_1[i][2])[1]).(1:length(xy1_1))
    val_gradf2 = (i -> gradf.(xy1_2[i][1], xy1_2[i][2])[2]).(1:length(xy1_2))
    @show maximum(abs.( Δy * colmat1_1\val_gradf1 -  gradfh[1]))
    @show maximum(abs.( Δx * colmat1_2\val_gradf2 -  gradfh[2]))

    val_g1 = (i -> g.(xy1_1[i][1], xy1_1[i][2])[1]).(1:length(xy1_1))
    val_g2 = (i -> g.(xy1_2[i][1], xy1_2[i][2])[2]).(1:length(xy1_2))
    @show maximum(abs.(Δy *colmat1_1\val_g1 - gh[1]))
    @show maximum(abs.(Δx *colmat1_2\val_g2 - gh[2]))

    val_curlg = (i -> curlg.(xy2[i][1], xy2[i][2])).(1:length(xy2))
    @show maximum(abs.(colmat2\val_curlg - curlgh))

    print("\nMultipatch Periodic B-Spline Basis: \n")
    X, Y = SplineFEEC.PeriodicMultipatchTensorProductBSpline(domain, k, N, el).B
    V0 = X ⊗ Y 
    V1 = SVector(X' ⊗ Y , X ⊗ Y')
    V2 = X' ⊗ Y' 

    G = disc_diff(grad, V0)
    C = disc_diff(curl, V0)

    xy0, colmat0 = col_mat(V0, 100)

    #need a mixed setup
    x, cmx = col_mat_d(X', 100)
    y, cmy = col_mat(Y, 100)
    xy1_1 = [(a,b) for a in x for b in y]
    colmat1_1 = cmx ⊗ cmy 

    x, cmx = col_mat(X, 100)
    y, cmy = col_mat_d(Y', 100)
    xy1_2 = [(a,b) for a in x for b in y]
    colmat1_2 = cmx ⊗ cmy 

    xy2, colmat2 = col_mat_d(V2, 100)

    fh = L2_proj(f, V0)

    #also mixed setup TODO introduce better functions
    M1x = mass_matrix_d(X')
    M1y = mass_matrix(Y)
    M1_1 = M1x ⊗ M1y
    gradfh1 = M1_1\SplineFEEC.L2_prod_mixed_x((x,y) -> gradf(x,y)[1], X' ⊗ Y)

    M2x = mass_matrix(X)
    M2y = mass_matrix_d(Y')
    M1_2 = M2x ⊗ M2y
    gradfh2 = M1_2\SplineFEEC.L2_prod_mixed_y((x,y) -> gradf(x,y)[2], X ⊗ Y')
    gradfh = SVector(gradfh1, gradfh2)
    @show maximum(abs.(vcat((G * fh - gradfh)...)))

    val_f = (i -> f.(xy0[i][1], xy0[i][2])).(1:length(xy0))
    @show maximum(abs.(colmat0 * fh - val_f))

    val_gradf1 = (i -> gradf.(xy1_1[i][1], xy1_1[i][2])[1]).(1:length(xy1_1))
    val_gradf2 = (i -> gradf.(xy1_2[i][1], xy1_2[i][2])[2]).(1:length(xy1_2))

    @show maximum(abs.(colmat1_1 * gradfh[1] - val_gradf1))
    @show maximum(abs.(colmat1_2 * gradfh[2] - val_gradf2))

    gh = SVector( M1_1\SplineFEEC.L2_prod_mixed_x((x,y) -> g(x,y)[1], X' ⊗ Y),
                  M1_2\SplineFEEC.L2_prod_mixed_y((x,y) -> g(x,y)[2], X ⊗ Y'))


    curlgh = L2_proj_d(curlg, V2)
    @show maximum(abs.((C * gh - curlgh)))

    val_g1 = (i -> g.(xy1_1[i][1], xy1_1[i][2])[1]).(1:length(xy1_1))
    val_g2 = (i -> g.(xy1_2[i][1], xy1_2[i][2])[2]).(1:length(xy1_2))
    @show maximum(abs.(colmat1_1 * gh[1] - val_g1))
    @show maximum(abs.(colmat1_2 * gh[2] - val_g2))
      
    val_curlg = (i -> curlg.(xy2[i][1], xy2[i][2])).(1:length(xy2))
    @show maximum(abs.(colmat2 * curlgh - val_curlg))

    #test collocation projections
    xy0, colmat0 = SplineFEEC.col_mat_proj(V0)
    xy1_1, colmat1_1 = SplineFEEC.col_mat_proj_mixed_x(V1[1])
    xy1_2, colmat1_2 = SplineFEEC.col_mat_proj_mixed_y(V1[2]) 
    xy2, colmat2 = SplineFEEC.col_mat_proj_d(V2)

    val_f = (i -> f.(xy0[i][1], xy0[i][2])).(1:length(xy0))
    @show maximum(abs.(colmat0\val_f - fh))

    val_gradf1 = (i -> gradf.(xy1_1[i][1], xy1_1[i][2])[1]).(1:length(xy1_1))
    val_gradf2 = (i -> gradf.(xy1_2[i][1], xy1_2[i][2])[2]).(1:length(xy1_2))
    @show maximum(abs.(colmat1_1\val_gradf1 - gradfh[1]))
    @show maximum(abs.(colmat1_2\val_gradf2 - gradfh[2]))

    val_g1 = (i -> g.(xy1_1[i][1], xy1_1[i][2])[1]).(1:length(xy1_1))
    val_g2 = (i -> g.(xy1_2[i][1], xy1_2[i][2])[2]).(1:length(xy1_2))
    @show maximum(abs.(colmat1_1\val_g1 - gh[1]))
    @show maximum(abs.(colmat1_2\val_g2 - gh[2]))

    val_curlg = (i -> curlg.(xy2[i][1], xy2[i][2])).(1:length(xy2))
    @show maximum(abs.(colmat2\val_curlg - curlgh))

    print("\nConga Periodic B-Spline Basis: \n")

    Δx = (domain.domains[1].right - domain.domains[1].left)/el[1]
    Δy = (domain.domains[2].right - domain.domains[2].left)/el[2]
   
    X, Y = SplineFEEC.PeriodicMultipatchTensorProductBSpline(domain, k, N, el).B
    V0 = X ⊗ Y 
    V1 = SVector(X' ⊗ Y , X ⊗ Y')
    V2 = X' ⊗ Y' 

    G = SplineFEEC.disc_diff(grad, V0)
    C = SplineFEEC.disc_diff(curl, V0)
    P = SplineFEEC.conforming_averaging_projection_matrix(V0)
    Px = SplineFEEC.conforming_averaging_projection_matrix_x(V0)
    Py = SplineFEEC.conforming_averaging_projection_matrix_y(V0)

    xy0, colmat0 = col_mat(V0, 100)
    xy1_1, colmat1_1 = SplineFEEC.col_mat_mixed_x(V1[1], 100) 
    xy1_2, colmat1_2 = SplineFEEC.col_mat_mixed_y(V1[2], 100) 
    xy2, colmat2 = col_mat_d(V2, 100)

    fh = L2_proj_d(f, V0)
    fh *= 1/(Δx*Δy)
    
    gradfh = L2_proj_d(gradf, V1)
    gradfh = SVector(1/Δy *gradfh[1],1/Δx * gradfh[2])
    @show maximum(abs.(vcat(( G * SVector{size(P)[1]}(P * fh) - SVector(Py * gradfh[1],Px * gradfh[2]) )...)))

    val_f = (i -> f.(xy0[i][1], xy0[i][2])).(1:length(xy0))
    @show maximum(abs.(colmat0 * P * fh - val_f))

    val_gradf1 = (i -> gradf.(xy1_1[i][1], xy1_1[i][2])[1]).(1:length(xy1_1))
    val_gradf2 = (i -> gradf.(xy1_2[i][1], xy1_2[i][2])[2]).(1:length(xy1_2))
    @show maximum(abs.(colmat1_1 * Py * gradfh[1] - val_gradf1))
    @show maximum(abs.(colmat1_2 * Px * gradfh[2] - val_gradf2))

    gh = L2_proj_d(g, V1)
    gh = SVector(1/Δy * gh[1], 1/Δx * gh[2])
    curlgh = L2_proj_d(curlg, V2)
    @show maximum(abs.((C * SVector(Py * gh[1], Px * gh[2] ) - curlgh)))

    val_g1 = (i -> g.(xy1_1[i][1], xy1_1[i][2])[1]).(1:length(xy1_1))
    val_g2 = (i -> g.(xy1_2[i][1], xy1_2[i][2])[2]).(1:length(xy1_2))
    @show maximum(abs.(colmat1_1 * Py * gh[1] - val_g1))
    @show maximum(abs.(colmat1_2 * Px * gh[2] - val_g2))
      
    val_curlg = (i -> curlg.(xy2[i][1], xy2[i][2])).(1:length(xy2))
    @show maximum(abs.(colmat2 * curlgh - val_curlg))

    #testing the colocation projection
    xy0, colmat0 = SplineFEEC.col_mat_proj_d(V0)
    xy1_1, colmat1_1 = SplineFEEC.col_mat_proj_d(V1[1])
    xy1_2, colmat1_2 = SplineFEEC.col_mat_proj_d(V1[2]) 
    xy2, colmat2 = SplineFEEC.col_mat_proj_d(V2)

    val_f = (i -> f.(xy0[i][1], xy0[i][2])).(1:length(xy0))
    @show maximum(abs.((Δx*Δy) * colmat0\val_f - fh))

    val_gradf1 = (i -> gradf.(xy1_1[i][1], xy1_1[i][2])[1]).(1:length(xy1_1))
    val_gradf2 = (i -> gradf.(xy1_2[i][1], xy1_2[i][2])[2]).(1:length(xy1_2))
    @show maximum(abs.( Δy * colmat1_1\val_gradf1 -  gradfh[1]))
    @show maximum(abs.( Δx * colmat1_2\val_gradf2 -  gradfh[2]))

    val_g1 = (i -> g.(xy1_1[i][1], xy1_1[i][2])[1]).(1:length(xy1_1))
    val_g2 = (i -> g.(xy1_2[i][1], xy1_2[i][2])[2]).(1:length(xy1_2))
    @show maximum(abs.(Δy *colmat1_1\val_g1 - gh[1]))
    @show maximum(abs.(Δx *colmat1_2\val_g2 - gh[2]))

    val_curlg = (i -> curlg.(xy2[i][1], xy2[i][2])).(1:length(xy2))
    @show maximum(abs.(colmat2\val_curlg - curlgh))
end

function test_Poisson_1d(k::Int, N::Int, el::Int)
    u(x) = sin(2π * x)
    f(x) = -4π^2 * sin(2π * x)
    interval = 0..1
    
    print("Periodic B-Spline Basis: \n")
    basis = PeriodicBSpline(interval, k, N)

    Δ = stiffness_matrix(basis') 
    R = 1/N .* circ_ones(N)
    P = circ_id(N) - R
    fh = L2_prod(f, basis)
    uh = -(Δ + R) \ (P * fh)

    x, colmat = col_mat(basis, 200)
    @show maximum( abs.(colmat * uh .- u.(x) )) 

    print("Dirichlet B-Spline Basis: \n")
    basis = DirichletBSpline(interval, k, N)

    fh = L2_prod(f, basis)
    # enforce homogenous boundary conditions
    Δ = stiffness_matrix(basis')  
    ϵ = 1e-12
    Δ[1,1] *= (1 + 1/ϵ) 
    Δ[end,end] *= (1 + 1/ϵ) 
    
    uh = -Δ \ fh
    
    x, colmat = col_mat(basis, 200)
    @show maximum( abs.(colmat * uh .- u.(x) )) 
    
    print("Multi-patch Dirichlet B-Spline Basis: \n")
    basis = SplineFEEC.DirichletMultipatchBSpline(interval, k, N, el)
    
    fh = L2_prod(f, basis)
    Δ = stiffness_matrix(basis')
    #dirichlet bc
    ϵ = 1e-12
    Δ[1,1] *= (1 +1/ϵ) 
    Δ[end,end] *= (1 +1/ϵ) 

    uh = -Δ \ fh
    
    x, colmat = col_mat(basis, 200)  
    @show maximum( abs.(colmat * uh .- u.(x) )) 
end

function test_Maxwells_1d(k::Int, N::Int, el::Int)
    # exact solution 
    B = (t, x) -> cos(2π * t) * cos(2π * x)
    E = (t, x) -> sin(2π * t) * sin(2π * x)
    J = (t, x) -> 0
    domain = (0..1)
    it = 1234
    Δt = 0.001
    T = Δt * it
    Δx = (domain.right - domain.left)/el

    @show (k, N, el)
    @show Δt, it 

    print("\nPeriodic B-Spline Basis: \n")
    # define the bases
    X = PeriodicBSpline(domain, k, N)
    V0 = X 
    V1 = X'

    # calculate the mass matrices
    M0 = mass_matrix(V0) 
    M0inv = inv(M0) # using the circulant inverse
    M1 = mass_matrix(V1)

    # calculate the discrete derivative
    G = disc_grad(V0)

    # discrete projections
    Bh = L2_proj(x -> B(0, x), V1)
    Eh = L2_proj(x -> E(0, x), V0)

    # simple leap frog scheme
    for i in 1:it
        Eh += Δt/2 * M0inv * (G' * (M1 * Bh)) 
        Bh -= Δt * (G * Eh)
        Eh += Δt/2 * M0inv * (G' * (M1 * Bh))
    end

    # collocation matrices
    x0, cm_0 = col_mat(V0, 100)
    x1, cm_1 = col_mat(V1, 100)

    @show maximum(abs.(cm_0 * Eh - E.(T, x0)))
    @show maximum(abs.(cm_1 * Bh - B.(T, x1)))

    print("\nDirichlet B-Spline Basis: \n")
    X = DirichletBSpline(domain, k, N)
    V0 = X 
    V1 = X'

    # calculate the mass matrices
    M0 = mass_matrix(V0) 

    ϵ = 1e-14
    M0[1,1] *= (1 + 1/ϵ)
    M0[end,end] *= (1 + 1/ϵ)

    M0inv = inv(M0) # using the circulant inverse
    M1 = mass_matrix(V1)

    # calculate the discrete derivative
    G = disc_grad(V0)

    # discrete projections
    Bh = L2_proj(x -> B(0, x), V1)
    Eh = L2_proj(x -> E(0, x), V0)

    # simple leap frog scheme
    for i in 1:it
        Eh += Δt/2 * M0inv * (G' * (M1 * Bh)) 
        Bh -= Δt * (G * Eh)
        Eh += Δt/2 * M0inv * (G' * (M1 * Bh))
    end

    # collocation matrices
    x0, cm_0 = col_mat(V0, 100)
    x1, cm_1 = col_mat(V1, 100)

    @show maximum(abs.(cm_0 * Eh - E.(T, x0)))
    @show maximum(abs.(cm_1 * Bh - B.(T, x1)))

    print("\nMulti-patch Dirichlet B-Spline Basis: \n")
    X = DirichletMultipatchBSpline(domain, k, N, el)
    V0 = X 
    V1 = X'

    # calculate the mass matrices
    M0 = mass_matrix(V0) 

    ϵ = 1e-14
    M0[1,1] *= (1 + 1/ϵ)
    M0[end,end] *= (1 + 1/ϵ)

    M0inv = inv(M0) # using the circulant inverse
    M1 = mass_matrix_d(V1) #special mass matrix

    # calculate the discrete derivative
    G = disc_grad(V0)

    # discrete projections
    Eh = L2_proj(x -> E(0, x), V0)
    Bh = L2_proj_d(x -> B(0, x), V1)

    # simple leap frog scheme
    for i in 1:it
        Eh += Δt/2 * M0inv * (G' * (M1 * Bh)) 
        Bh -= Δt * (G * Eh)
        Eh += Δt/2 * M0inv * (G' * (M1 * Bh))
    end

    # collocation matrices
    x0, cm_0 = col_mat(V0, 100)
    x1, cm_1 = col_mat_d(V1, 100)

    @show maximum(abs.(cm_0 * Eh - E.(T, x0)))
    @show maximum(abs.(cm_1 * Bh - B.(T, x1)))

    print("\nConga Dirichlet B-Spline Basis: \n")
    X = DirichletMultipatchBSpline(domain, k, N, el)
    V0 = X 
    V1 = X'

    # calculate the mass matrices
    M0 = mass_matrix_d(V0) 

    ϵ = 1e-14
    M0[1,1] *= (1 + 1/ϵ)
    M0[end,end] *= (1 + 1/ϵ)

    M0inv = 1/Δx^2 * inv(M0) # using the circulant inverse
    M1 = mass_matrix_d(V1) #special mass matrix

    # calculate the discrete derivative
    G = disc_grad(V0)
    P = conforming_averaging_projection_matrix(V0)

    # discrete projections
    Eh = 1/Δx * L2_proj_d(x -> E(0, x), V0)
    Bh = L2_proj_d(x -> B(0, x), V1)

    # simple leap frog scheme
    for i in 1:it
        Eh += Δt/2 * M0inv * (P' * G' * (M1 * Bh)) 
        Bh -= Δt * (G * P * Eh)
        Eh += Δt/2 * M0inv * (P' * G' * (M1 * Bh))
    end

    # collocation matrices
    x0, cm_0 = col_mat(V0, 100)
    x1, cm_1 = col_mat_d(V1, 100)

    @show maximum(abs.(cm_0 * P * Eh - E.(T, x0)))
    @show maximum(abs.(cm_1 * Bh - B.(T, x1)))

    print("\nMulti-patch Periodic B-Spline Basis: \n")
    X = PeriodicMultipatchBSpline(domain, k, N, el)
    V0 = X 
    V1 = X'

    # calculate the mass matrices
    M0 = mass_matrix(V0) 

    M0inv = inv(M0) # using the circulant inverse
    M1 = mass_matrix_d(V1) #special mass matrix

    # calculate the discrete derivative
    G = disc_grad(V0)

    # discrete projections
    Eh = L2_proj(x -> E(0, x), V0)
    Bh = L2_proj_d(x -> B(0, x), V1)

    # simple leap frog scheme
    for i in 1:it
        Eh += Δt/2 * M0inv * (G' * (M1 * Bh)) 
        Bh -= Δt * (G * Eh)
        Eh += Δt/2 * M0inv * (G' * (M1 * Bh))
    end

    # collocation matrices
    x0, cm_0 = col_mat(V0, 100)
    x1, cm_1 = col_mat_d(V1, 100)

    @show maximum(abs.(cm_0 * Eh - E.(T, x0)))
    @show maximum(abs.(cm_1 * Bh - B.(T, x1)))

    print("\nConga Periodic B-Spline Basis: \n")
    X = SplineFEEC.PeriodicMultipatchBSpline(domain, k, N, el)
    V0 = X 
    V1 = X'

    # calculate the mass matrices
    M0 = mass_matrix_d(V0) 

    M0inv = 1/Δx^2 * inv(M0) # using the circulant inverse
    M1 = mass_matrix_d(V1) #special mass matrix

    # calculate the discrete derivative
    G = disc_grad(V0)
    P = SplineFEEC.conforming_averaging_projection_matrix(V0)

    # discrete projections
    Eh = 1/Δx * L2_proj_d(x -> E(0, x), V0)
    Bh = L2_proj_d(x -> B(0, x), V1)

    # simple leap frog scheme
    for i in 1:it
        Eh += Δt/2 * M0inv * (P' * G' * (M1 * Bh)) 
        Bh -= Δt * (G * P * Eh)
        Eh += Δt/2 * M0inv * (P' * G' * (M1 * Bh))
    end

    # collocation matrices
    x0, cm_0 = col_mat(V0, 100)
    x1, cm_1 = col_mat_d(V1, 100)

    @show maximum(abs.(cm_0 * P * Eh - E.(T, x0)))
    @show maximum(abs.(cm_1 * Bh - B.(T, x1)))
end

function test_Issautier_1d(k::Int, N::Int, el::Int)
    # exact solution 
    B = (t, x) -> (cos(t) - 1) * π * cos(π * x)
    E = (t, x) -> sin(t) * sin(π * x)
    J = (t, x) -> (cos(t) - 1) * π^2 * sin(π * x) - cos(t) * sin(π * x)
    domain = (0..1)
    it = 1234
    Δt = 0.01
    T = Δt * it
    Δx = (domain.right - domain.left)/el

    @show (k, N, el)
    @show Δt, it 

    #exact E L2-norm
    tt = 0
    Enex = zeros(it)
    for i in 1:it
        tt += Δt
        Enex[i] = sqrt(hquadrature(x -> E(tt, x).^2, 0, 1)[1])
    end

    print("\nDirichlet B-Spline Basis: \n")
    X = DirichletBSpline(domain, k, N)
    V0 = X 
    V1 = X'

    # calculate the mass matrices
    M0 = mass_matrix(V0) 
    M0_old = deepcopy(M0)

    ϵ = 1e-13
    M0[1,1] *= (1 + 1/ϵ)
    M0[end,end] *= (1 + 1/ϵ)

    M0inv = inv(M0) # using the circulant inverse
    M1 = mass_matrix(V1)

    # calculate the discrete derivative
    G = disc_grad(V0)

    # discrete projections
    Bh = L2_proj(x -> B(0, x), V1)
    Eh = L2_proj(x -> E(0, x), V0)
    Jh = L2_prod(x -> J(0,x), V0)

    En1 = zeros(it)
    # simple leap frog scheme
    t = 0
    for i in 1:it
        Eh += Δt/2 * M0inv * (G' * (M1 * Bh) - Jh) 

        Bh -= Δt * (G * Eh)
        t += Δt
        Jh =  L2_prod(x -> J(t, x), V0)

        Eh += Δt/2 * M0inv * (G' * (M1 * Bh) - Jh)
        En1[i] = 1/(2*X.N) * sqrt(dot(Eh, M0_old * Eh))
    end

    # collocation matrices
    x0, cm_0 = col_mat(V0, 100)
    x1, cm_1 = col_mat(V1, 100)

    @show maximum(abs.(cm_0 * Eh - E.(T, x0)))
    @show maximum(abs.(cm_1 * Bh - B.(T, x1)))
    @show maximum(abs.(En1 - Enex))

    print("\nMulti-patch Dirichlet B-Spline Basis: \n")
    X = DirichletMultipatchBSpline(domain, k, N, el)
    V0 = X 
    V1 = X'

    # calculate the mass matrices
    M0 = mass_matrix(V0) 
    M0_old = deepcopy(M0)

    ϵ = 1e-13
    M0[1,1] *= (1 + 1/ϵ)
    M0[end,end] *= (1 + 1/ϵ)

    M0inv = inv(M0) # using the circulant inverse
    M1 = mass_matrix_d(V1) #special mass matrix

    # calculate the discrete derivative
    G = disc_grad(V0)

    # discrete projections
    Eh = L2_proj(x -> E(0, x), V0)
    Bh = L2_proj_d(x -> B(0, x), V1)
    Jh = L2_prod(x -> J(0,x), V0)

    # simple leap frog scheme
    En2 = zeros(it)
    t = 0
    for i in 1:it
        Eh += Δt/2 * M0inv * (G' * (M1 * Bh) - Jh) 
        Bh -= Δt * (G * Eh)
        t += Δt
        Jh =  L2_prod(x -> J(t, x), V0)
        Eh += Δt/2 * M0inv * (G' * (M1 * Bh) - Jh)
        En2[i] = 1/(2*X.B.N*el) * sqrt(dot(Eh, M0_old * Eh))
    end

    # collocation matrices
    x0, cm_0 = col_mat(V0, 100)
    x1, cm_1 = col_mat_d(V1, 100)

    @show maximum(abs.(cm_0 * Eh - E.(T, x0)))
    @show maximum(abs.(cm_1 * Bh - B.(T, x1)))
    @show maximum(abs.(En2 - Enex))

    print("\nConga Dirichlet B-Spline Basis: \n")
    X = DirichletMultipatchBSpline(domain, k, N, el)
    V0 = X 
    V1 = X'

    # calculate the mass matrices
    M0 = mass_matrix_d(V0) 
    M0_old = deepcopy(M0)

    ϵ = 1e-13
    M0[1,1] *= (1 + 1/ϵ)
    M0[end,end] *= (1 + 1/ϵ)

    M0inv = 1/Δx^2 * inv(M0) # using the circulant inverse
    M1 = mass_matrix_d(V1) #special mass matrix

    # calculate the discrete derivative
    G = disc_grad(V0)
    P = conforming_averaging_projection_matrix(V0)

    # discrete projections
    Eh = 1/Δx * L2_proj_d(x -> E(0, x), V0)
    Bh = L2_proj_d(x -> B(0, x), V1)
    #we could also directly project J on V^0_b
    Jh = L2_prod(x -> J(0, x), V0)

    En3 = zeros(it)
    # simple leap frog scheme
    t = 0
    for i in 1:it
        Eh += Δt/2 * M0inv * P' * (G' * (M1 * Bh) -   Jh)
        Bh -= Δt * (G * P * Eh)
        
        t += Δt
        Jh =  L2_prod(x -> J(t, x), V0)
        
        Eh += Δt/2 * M0inv * P' * (G' * (M1 * Bh) - Jh)
        En3[i] = 1/(2*X.B.N*el^2) * sqrt(dot(Eh, M0_old * Eh))
    end

    # collocation matrices
    x0, cm_0 = col_mat(V0, 100)
    x1, cm_1 = col_mat_d(V1, 100)

    @show maximum(abs.(cm_0 * P * Eh - E.(T, x0)))
    @show maximum(abs.(cm_1 * Bh - B.(T, x1)))
    @show maximum(abs.(En3 - Enex))

    return it, En1, En2, En3, Enex
end

# -Δu = f
function test_Poisson_2d(k::NTuple{2,Int}, N::NTuple{2,Int})
    u(x, y) = sin(2π * x) * sin(2π * y) 
    f(x, y) = -4π^2 * sin(2π * x)* sin(2π * y) - 4π^2 * sin(2π * y) * sin(2π * x)
    domain = (0..1) × (0..1)

    print("Periodic B-Spline Basis: \n")
    basis = PeriodicBSplineTensorProductBasis(domain, k, N)

    Δ = ∇(basis) ⊙ ∇(basis)
    NN = prod(N)
    R = 1/NN .* SplineFEEC.circ_ones(NN)
    P = SplineFEEC.circ_id(NN) - R
    fh = L2_prod(f, basis)

    uh = -(Δ + R) \ (P * fh)

    xy, colmat = col_mat(basis, 100)
    val_u = (i -> u.(xy[i][1], xy[i][2])).(1:length(xy))
    @show maximum(abs.(val_u - colmat * uh))
    #return xy, colmat, fh, Δ, uh
    
    print("Dirichlet B-Spline Basis: \n")
    basis = DirichletBSplineTensorProductBasis(domain, k, N)

    #homogeneous bc built in
    Δ = ∇(basis) ⊙ ∇(basis)
    fh = L2_prod(f, basis)
    uh = -Δ\ fh

    xy, colmat = col_mat(basis, 100)
    val_u = (i -> u.(xy[i][1], xy[i][2])).(1:length(xy))
    @show maximum(abs.(val_u - colmat * uh))
end

function test_Maxwells_Issautier_2d(k::NTuple{2,Int}, N::NTuple{2,Int}, el::NTuple{2,Int})
    E₂(t, x, y) = sin(t) .* SVector( x * sin(π * y), y *  sin(π * x)) 
    B₂(t, x, y) = (cos(t) - 1) * π * ( y * cos(x*π) - x * cos(π * y)) 
    J₂(t, x, y) = (cos(t) - 1) .* SVector(π * cos(π * x) + π^2 * x * sin(π * y), π * cos(π * y) + π^2 * y * sin(π * x)) - cos(t) .* SVector( x * sin(π * y),  y * sin(π * x))
    domain = (0..1) × (0..1) 

    @show k
    @show N
    @show el

    it = 1234
    Δt = 0.01
    #it = 1234
    #Δt = 0.00001
    T = Δt * it
    @show Δt, it

    tt = 0
    Enex = zeros(it)
    for i in 1:it
        tt += Δt
        Enex[i] = sqrt(hcubature(x -> dot(E₂(tt, x[1], x[2]), E₂(tt, x[1], x[2])), (0, 0), (1,1))[1])
    end

    print("Dirichlet B-Spline Basis: \n")
    # define bases
    X,Y = DirichletBSplineTensorProductBasis(domain, k, N).B
    V0 = X ⊗ Y
    V1 = SVector(X' ⊗ Y, X ⊗ Y')
    V2 = X' ⊗ Y'

    # calculate mass matrices
    M1 = mass_matrix.(V1) #Vector representing the diagonal of the mass blockmatrix
    M1_old = deepcopy(M1)

    ϵ = 1e-12
    M1[1].B[1,1] *= (1+1/ϵ)
    M1[1].B[end,end] *= (1+1/ϵ)
    M1[2].A[1,1] *= (1+1/ϵ)
    M1[2].A[end,end] *= (1+1/ϵ)

    M1inv = inv.(M1) #using the circulant inverse

    M2 = mass_matrix(V2)

    # calculate discrete curl
    C = disc_curl(V0)

    # or by colocation projection
    (xy1_1, colmat1_1), (xy1_2, colmat1_2) = col_mat_proj.(V1)
    xy2, colmat2 = col_mat_proj(V2)

    cmp_inv1 = inv(colmat1_1)
    cmp_inv2 = inv(colmat1_2)

    val_B = (i -> B₂.(0, xy2[i][1], xy2[i][2])).(1:length(xy2))
    val_E1 = (i -> E₂.(0, xy1_1[i][1], xy1_1[i][2])[1]).(1:length(xy1_1))
    val_E2 = (i -> E₂.(0, xy1_2[i][1], xy1_2[i][2])[2]).(1:length(xy1_2));
    val_J1 = (i -> J₂.(0, xy1_1[i][1], xy1_1[i][2])[1]).(1:length(xy1_1))
    val_J2 = (i -> J₂.(0, xy1_2[i][1], xy1_2[i][2])[2]).(1:length(xy1_2));

    Eh = SVector(cmp_inv1 * val_E1, cmp_inv2 * val_E2)
    Jh = SVector(cmp_inv1 * val_J1, cmp_inv2 * val_J2)
    Bh = colmat2\val_B;

    #prepare time stepping
    t = 0

    En1 = zeros(it)
    for i in 1:it
        Eh += Δt/2 * (M1inv .* (C' *  SVector{length(Bh)}(M2 * Bh) - Jh))
        Bh -= Δt * (C * Eh)

        t += Δt
        val_J1 = (i -> J₂.(t, xy1_1[i][1], xy1_1[i][2])[1]).(1:length(xy1_1))
        val_J2 = (i -> J₂.(t, xy1_2[i][1], xy1_2[i][2])[2]).(1:length(xy1_2))
        Jh = M1_old .* SVector(cmp_inv1 * val_J1, cmp_inv2 * val_J2)

        Eh += Δt/2 * (M1inv .* (C' *  SVector{length(Bh)}(M2 * Bh) - Jh))
        En1[i] =  1/(2*X.N) * 1/(2*Y.N) * sqrt(dot(Eh, M1_old .* Eh))
    end

    (xy1_1, colmat1_1), (xy1_2, colmat1_2) = col_mat.(V1, 10)
    xy2, colmat2 = col_mat(V2, 10)
    val_B = (i -> B₂.(T, xy2[i][1], xy2[i][2])).(1:length(xy2))
    val_E1 = (i -> E₂.(T, xy1_1[i][1], xy1_1[i][2])[1]).(1:length(xy1_1))
    val_E2 = (i -> E₂.(T, xy1_2[i][1], xy1_2[i][2])[2]).(1:length(xy1_2))

    @show maximum(abs.(colmat1_1 * Eh[1] - val_E1))
    @show maximum(abs.(colmat1_2 * Eh[2] - val_E2))
    @show maximum(abs.(colmat2 * Bh - val_B))
    @show maximum(abs.(En1 - Enex))

    print("Multipatch BSpline Basis:\n")
    # define bases
    X,Y = DirichletMultipatchTensorProductBSpline(domain, k, N, el).B
    V0 = X ⊗ Y
    V1 = SVector(X' ⊗ Y, X ⊗ Y')
    V2 = X' ⊗ Y'

    ϵ = 1e-12
    M1x = mass_matrix_d(X')
    M1y = mass_matrix(Y)
    M1_1_old = deepcopy(M1x ⊗ M1y)

    M1y[1,1] *= (1+1/ϵ)
    M1y[end,end] *= (1+1/ϵ)
    M1_1 = M1x ⊗ M1y
    
    M2x = mass_matrix(X)
    M2y = mass_matrix_d(Y')
    M1_2_old = deepcopy(M2x ⊗ M2y)

    M2x[1,1] *= (1+1/ϵ)
    M2x[end,end] *= (1+1/ϵ)
    M1_2 = M2x ⊗ M2y
    
    M1 = SVector(M1_1, M1_2)
    M1_old = SVector(M1_1_old, M1_2_old)

    M1inv = inv.(M1) #using the circulant inverse
    
    M2 = mass_matrix_d(V2)

    # calculate discrete curl
    C = disc_diff(curl, V0)
    
    # colocation projection
    xy1_1, colmat1_1 = SplineFEEC.col_mat_proj_mixed_x(V1[1])
    xy1_2, colmat1_2 = SplineFEEC.col_mat_proj_mixed_y(V1[2]) 
    xy2, colmat2 = SplineFEEC.col_mat_proj_d(V2)

    cmp_inv1 = inv(colmat1_1)
    cmp_inv2 = inv(colmat1_2)

    val_B = (i -> B₂.(0, xy2[i][1], xy2[i][2])).(1:length(xy2))
    val_E1 = (i -> E₂.(0, xy1_1[i][1], xy1_1[i][2])[1]).(1:length(xy1_1))
    val_E2 = (i -> E₂.(0, xy1_2[i][1], xy1_2[i][2])[2]).(1:length(xy1_2));
    val_J1 = (i -> J₂.(0, xy1_1[i][1], xy1_1[i][2])[1]).(1:length(xy1_1))
    val_J2 = (i -> J₂.(0, xy1_2[i][1], xy1_2[i][2])[2]).(1:length(xy1_2));

    Eh = SVector(cmp_inv1 * val_E1, cmp_inv2 * val_E2)
    Jh = SVector(cmp_inv1 * val_J1, cmp_inv2 * val_J2)
    Bh = colmat2\val_B;

    #prepare time stepping
    t = 0
    En2 = zeros(it)
    for i in 1:it
        Eh += Δt/2 * (M1inv .* (C' *  SVector{length(Bh)}(M2 * Bh) - Jh))
        Bh -= Δt * (C * Eh)

        t += Δt
        val_J1 = (i -> J₂.(t, xy1_1[i][1], xy1_1[i][2])[1]).(1:length(xy1_1))
        val_J2 = (i -> J₂.(t, xy1_2[i][1], xy1_2[i][2])[2]).(1:length(xy1_2))
        Jh = M1_old .* SVector(cmp_inv1 * val_J1, cmp_inv2 * val_J2)

        Eh += Δt/2 * (M1inv .* (C' *  SVector{length(Bh)}(M2 * Bh) - Jh))
        En2[i] =  1/(2*X.B.N*el[1]) * 1/(2 * Y.B.N*el[2]) * sqrt(dot(Eh, M1_old .* Eh))
    end

    
    x, cmx = col_mat_d(X', 10)
    y, cmy = col_mat(Y, 10)
    xy1_1 = [(a,b) for a in x for b in y]
    colmat1_1 = cmx ⊗ cmy 

    x, cmx = col_mat(X, 10)
    y, cmy = col_mat_d(Y', 10)
    xy1_2 = [(a,b) for a in x for b in y]
    colmat1_2 = cmx ⊗ cmy 

    xy2, colmat2 = col_mat_d(V2, 10)
    
    val_B = (i -> B₂.(T, xy2[i][1], xy2[i][2])).(1:length(xy2))
    val_E1 = (i -> E₂.(T, xy1_1[i][1], xy1_1[i][2])[1]).(1:length(xy1_1))
    val_E2 = (i -> E₂.(T, xy1_2[i][1], xy1_2[i][2])[2]).(1:length(xy1_2))

    @show maximum(abs.(colmat1_1 * Eh[1] - val_E1))
    @show maximum(abs.(colmat1_2 * Eh[2] - val_E2))
    @show maximum(abs.(colmat2 * Bh - val_B))
    @show maximum(abs.(En2 - Enex))

    print("Conga Multipatch BSpline Basis:\n")
    Δx = (domain.domains[1].right - domain.domains[1].left)/el[1]
    Δy = (domain.domains[2].right - domain.domains[2].left)/el[2]
    
    # define bases
    X,Y = SplineFEEC.DirichletMultipatchTensorProductBSpline(domain, k, N, el).B
    V0 = X ⊗ Y
    V1 = SVector(X' ⊗ Y, X ⊗ Y')
    V2 = X' ⊗ Y'
    
    M1 = mass_matrix_d.(V1)
    M1_old = deepcopy(M1)

    ϵ = 1e-12
    M1[1].B[1,1] *= (1+1/ϵ)
    M1[1].B[end,end] *= (1+1/ϵ)
    M1[2].A[1,1] *= (1+1/ϵ)
    M1[2].A[end,end] *= (1+1/ϵ)
    
    M1inv = inv.(M1)
    M1inv = SVector(1/(Δy)^2, 1/(Δx)^2) .* M1inv

    
    M2 = mass_matrix_d(V2)

    # calculate discrete curl
    C = disc_diff(curl, V0)
    
    P = SplineFEEC.conforming_averaging_projection_matrix(V0)
    Px = SplineFEEC.conforming_averaging_projection_matrix_x(V0)
    Py = SplineFEEC.conforming_averaging_projection_matrix_y(V0)

    
    # or by colocation projection
    (xy1_1, colmat1_1), (xy1_2, colmat1_2) = SplineFEEC.col_mat_proj_d.(V1)
    xy2, colmat2 = SplineFEEC.col_mat_proj_d(V2)

    cmp_inv1 = inv(colmat1_1)
    cmp_inv2 = inv(colmat1_2)

    val_B = (i -> B₂.(0, xy2[i][1], xy2[i][2])).(1:length(xy2))
    val_E1 = (i -> E₂.(0, xy1_1[i][1], xy1_1[i][2])[1]).(1:length(xy1_1))
    val_E2 = (i -> E₂.(0, xy1_2[i][1], xy1_2[i][2])[2]).(1:length(xy1_2));
    val_J1 = (i -> J₂.(0, xy1_1[i][1], xy1_1[i][2])[1]).(1:length(xy1_1))
    val_J2 = (i -> J₂.(0, xy1_2[i][1], xy1_2[i][2])[2]).(1:length(xy1_2));

    Eh = SVector(Δy *cmp_inv1 * val_E1, Δx *Δy *cmp_inv2 * val_E2)
    Jh = SVector(Δy *cmp_inv1 * val_J1, Δx *cmp_inv2 * val_J2)
    Bh = colmat2\val_B;
    
    
    #prepare time stepping
    t = 0
    En3 = zeros(it)
    for i in 1:it
        Eh += Δt/2 *  (M1inv .* (SVector(Py', Px') .* (C' *  SVector{length(Bh)}(M2 * Bh)) - Jh))
        Bh -= Δt *  (C * (SVector(Py , Px) .* Eh))

        t += Δt
        val_J1 = (i -> J₂.(t, xy1_1[i][1], xy1_1[i][2])[1]).(1:length(xy1_1))
        val_J2 = (i -> J₂.(t, xy1_2[i][1], xy1_2[i][2])[2]).(1:length(xy1_2))
        Jh =  M1_old .* SVector(Δy * cmp_inv1 * val_J1, Δx * cmp_inv2 * val_J2)

        Eh += Δt/2 *  (M1inv .* (SVector(Py', Px') .* (C' *  SVector{length(Bh)}(M2 * Bh)) - Jh))
        En3[i] =  1/(2*X.B.N*el[1]^2) * 1/(Y.B.N*el[2]^2) * sqrt(dot(Eh, M1_old .* Eh))
    end

    
    xy1_1, colmat1_1 = SplineFEEC.col_mat_mixed_x(V1[1], 10) 
    xy1_2, colmat1_2 = SplineFEEC.col_mat_mixed_y(V1[2], 10) 
    xy2, colmat2 = SplineFEEC.col_mat_d(V2, 10)
    
    val_B = (i -> B₂.(T, xy2[i][1], xy2[i][2])).(1:length(xy2))
    val_E1 = (i -> E₂.(T, xy1_1[i][1], xy1_1[i][2])[1]).(1:length(xy1_1))
    val_E2 = (i -> E₂.(T, xy1_2[i][1], xy1_2[i][2])[2]).(1:length(xy1_2))

    @show maximum(abs.(colmat1_1 * Py * Eh[1] - val_E1))
    @show maximum(abs.(colmat1_2 * Px * Eh[2] - val_E2))
    @show maximum(abs.(colmat2 * Bh - val_B))
    @show maximum(abs.(En3 - Enex))
    return it, En1, En2, En3, Enex
end


function test_cold_plasma_xwave_2d(k::NTuple{2,Int}, N::NTuple{2,Int}, el::NTuple{2,Int}, it, Δt)

    #todo fill in for energy conservation
    w_c = 2
    w_p = 1
    w = 0.5

    R = 1 - w_p^2/(w*(w-w_c))
    L = 1 - w_p^2/(w*(w+w_c))
    S = 1/2 * (R + L)

    #Langmuir test
    k2 = 0
    k1 = w * sqrt(R*L/S)
    E(t, x, y) = real.(SVector(1, -im * (R+L)/(R-L)) .* exp(-im*(w*t -k1*x -k2*y)))
    B(t, x, y) = real(-k1/w * im * (R+L)/(R-L) * exp(-im*(w*t -k1*x -k2*y)) )
    J(t, x, y) = real.( w_p^2 .* SVector(im * 1/(w+w_c), 1/(w-w_c) * (R+L)/(R-L)) .* exp(-im*(w*t -k1*x -k2*y)))

    E0(x,y) = E(0, x, y)
    B0(x,y) = B(0, x, y)
    J0(x,y) = J(0, x, y)

    ω_c(x,y) = w_c
    ω_p(x,y) = w_p

    domain = (0..2π/k1) × (0..2π) 
    T = Δt * it
    Δx = (domain.domains[1].right - domain.domains[1].left)/el[1]
    Δy = (domain.domains[2].right - domain.domains[2].left)/el[2]

    # Periodic Single Patch
    X,Y = PeriodicBSplineTensorProductBasis(domain, k, N).B
    V0 = X ⊗ Y
    V1 = SVector(X' ⊗ Y, X ⊗ Y')
    V2 = X' ⊗ Y'

    M1 = mass_matrix.(V1) #Vector representing the diagonal of the mass blockmatrix
    M2 = mass_matrix(V2)
    M1inv = inv.(M1)

    C = disc_curl(V0)
    Cc = disc_grad(V2).D[1];
    Cc2 = disc_grad(V1[2]).D[1];

    Eh = SplineFEEC.L2_proj(E0, V1)
    Bh = SplineFEEC.L2_proj(B0, V2)
    Jh = SplineFEEC.L2_proj(J0, V1)

    (xy1_1, colmat1_1), (xy1_2, colmat1_2) = SplineFEEC.col_mat.(V1, 50)
    xy2, colmat2 = SplineFEEC.col_mat(V2, 50);

    t = 0;
    #keep track of energy
    En_E = []
    En_B = []
    En_J = []
    En = []
    Enex_E = []
    Enex_J = []
    Enex_B = []
    Enex = [];

    M_p = w_p .* mass_matrix.(V1)
    M_c = w_c * mass_matrix(X', X ) ⊗ mass_matrix(Y,  Y');

    A1 = [2/Δt .* M1[1]  zero(M1[2]) zero(M1[1]);
    zero(M1[1]) 2/Δt .* M1[2]  -1/2*Cc'*M2;
    zero(M1[1])  1/2*Cc2  2/Δt .* I ]

    A2 = [2/Δt .*M1[1] zero(M1[2]) zero(M1[1]);
    zero(M1[1]) 2/Δt .* M1[2]  1/2*Cc'*M2;
    zero(M1[1]) -1/2*Cc2  2/Δt .* I ];

    B11 = [1/Δt .* M1[1] zero(M1[2]) 1/2 .* M1[1] zero(M1[2]);
            zero(M1[1])  1/Δt .* M1[2] zero(M1[2]) 1/2 .* M1[2];
            -M1[1] * w_p^2/2 zero(M1[2]) 1/Δt*M1[1]  w_c/2*M_c;
            zero(M1[1]) -M1[2]*w_p^2/2  -w_c/2*M_c' 1/Δt*M1[2]]

    B2 = [1/Δt .* M1[1] zero(M1[2]) -1/2 .* M1[1] zero(M1[2]);
            zero(M1[1])  1/Δt .* M1[2] zero(M1[2]) -1/2 .* M1[2];
            M1[1] * w_p^2/2 zero(M1[2]) 1/Δt*M1[1]  -w_c/2*M_c;
            zero(M1[1]) M1[2]*w_p^2/2  w_c/2*M_c' 1/Δt*M1[2]];

    A1inv = inv(A1)
    B1inv = inv(B11);

    Eh1 = Eh[1]
    Eh2 = Eh[2]
    Jh1 = Jh[1]
    Jh2 = Jh[2]
    Bh = Bh

    U1 = [Eh1; Eh2; Bh] 
    U2 = [Eh1; Eh2; Jh1; Jh2]
    for i in 1:it
            U1 = [Eh1; Eh2; Bh] 
            U1 = A1inv * (A2 * U1)

            Eh1 = U1[1:length(Eh1)]
            Eh2 = U1[length(Eh1)+1:length(Eh1)+length(Eh2)]
            Bh = U1[length(Eh1)+length(Eh2)+1:end]

            U2 = [Eh1; Eh2; Jh1; Jh2]
            U2 = B1inv * (B2 * U2)

            Eh1 = U2[1:length(Eh1)]
            Eh2 = U2[length(Eh1)+1:length(Eh1)+length(Eh2)]
            Jh1 = U2[length(Eh1)+length(Eh2)+1:length(Eh1)+length(Eh2)+length(Jh1)]
            Jh2 = U2[length(Eh1)+length(Eh2)+length(Jh1)+1:end]

            U1 = [Eh1; Eh2; Bh] 
            U1 = A1inv * (A2 * U1)

            Eh1 = U1[1:length(Eh1)]
            Eh2 = U1[length(Eh1)+1:length(Eh1)+length(Eh2)]
            Bh = U1[length(Eh1)+length(Eh2)+1:end]

            t += Δt
            push!(En_E, (2π/k1)/(2*X.N) * 2π/(2*Y.N) * sqrt.(Eh1'*M1[1]*Eh1 + Eh2'*M1[2]*Eh2))
            push!(En_J, (2π/k1)/(2*X.N) * 2π/(2*Y.N) * sqrt.(Jh1'*M1[1]*Jh1 + Jh2'*M1[2]*Jh2))
            push!(En_B, (2π/k1)/(2*X.N) * 2π/(2*Y.N) * sqrt.(Bh'*M2*Bh))
            push!(En, 1/2*En_E[end]^2 + 1/2*En_J[end]^2 + 1/2*En_B[end]^2 )

            push!(Enex_E, sqrt(hcubature(x -> dot(E(t, x[1], x[2]), E(t, x[1], x[2])), (0, 0), (2π/k1,2π))[1]))
            push!(Enex_J, sqrt(hcubature(x -> dot(J(t, x[1], x[2]), J(t, x[1], x[2])), (0, 0), (2π/k1,2π))[1]))
            push!(Enex_B, sqrt(hcubature(x -> dot(B(t, x[1], x[2]), B(t, x[1], x[2])), (0, 0), (2π/k1,2π))[1]))
            push!(Enex,  1/2*Enex_E[end]^2 + 1/2*Enex_B[end]^2 + 1/2*Enex_J[end]^2  )
    end

    Eh = SVector(Eh1, Eh2)
    Jh = SVector(Jh1, Jh2)
    Bh = Bh;

    t_n = t
    val_B = (i -> B.(t_n, xy2[i][1], xy2[i][2])).(1:length(xy2))
    val_E1 = (i -> E.(t_n, xy1_1[i][1], xy1_1[i][2])[1]).(1:length(xy1_1))
    val_E2 = (i -> E.(t_n, xy1_2[i][1], xy1_2[i][2])[2]).(1:length(xy1_2))
    val_J1 = (i -> J.(t_n, xy1_1[i][1], xy1_1[i][2])[1]).(1:length(xy1_1))
    val_J2 = (i -> J.(t_n, xy1_2[i][1], xy1_2[i][2])[2]).(1:length(xy1_2))

    err_per_E = max(maximum(abs.(colmat1_1 * Eh[1] - val_E1)), maximum(abs.(colmat1_2 * Eh[2] - val_E2)))
    err_per_J = max(maximum(abs.(colmat1_1 * Jh[1] - val_J1)), maximum(abs.(colmat1_2 * Jh[2] - val_J2)))
    err_per_B = maximum(abs.(colmat2 * Bh - val_B))
    En_per = En
    Enex_per = Enex


    #Periodic Multipatch
    X,Y = SplineFEEC.PeriodicMultipatchTensorProductBSpline(domain, k, N, el).B
            V0 = X ⊗ Y
            V1 = SVector(X' ⊗ Y, X ⊗ Y')
            V2 = X' ⊗ Y'


            M1x = mass_matrix_d(X')
            M1y = mass_matrix(Y)
            M1_1 = M1x ⊗ M1y

            M2x = mass_matrix(X)
            M2y = mass_matrix_d(Y')
            M1_2 = M2x ⊗ M2y

            M1 = SVector(M1_1, M1_2)
            M1inv = inv.(M1)

            M2 = mass_matrix_d(V2)


            Cc = SplineFEEC.disc_diff(curl, V0).D[1]
            Cc2 = disc_diff(curl, V0).D[1]

            Eh = SVector( M1_1\SplineFEEC.L2_prod_mixed_x((x,y) -> E0(x,y)[1], X' ⊗ Y),
                              M1_2\SplineFEEC.L2_prod_mixed_y((x,y) -> E0(x,y)[2], X ⊗ Y'))
            Jh = SVector( M1_1\SplineFEEC.L2_prod_mixed_x((x,y) -> J0(x,y)[1], X' ⊗ Y),
                              M1_2\SplineFEEC.L2_prod_mixed_y((x,y) -> J0(x,y)[2], X ⊗ Y'))
            Bh = SplineFEEC.L2_proj_d(B0, V2)


            xy0, colmat0 = col_mat(V0, 30)
            xy1_1, colmat1_1 = SplineFEEC.col_mat_mixed_x(V1[1], 30) 
            xy1_2, colmat1_2 = SplineFEEC.col_mat_mixed_y(V1[2], 30) 
            xy2, colmat2 = col_mat_d(V2, 30)

            t = 0;
            En_E = []
            En_B = []
            En_J = []
            En = []
            #Enex_E = []
            #Enex_J = []
            #Enex_B = []
            #Enex = [];
    
            M_p = w_p^2 .* M1
            M_c = w_c .* SplineFEEC.mixed_mass_matrix_x(V1[1], V1[2])
            
            A1 = [2/Δt .* M1[1]  zero(M1[2]) zero(M1[1]);
                    zero(M1[1]) 2/Δt .* M1[2]  -1/2*Cc'*M2;
                    zero(M1[1])  1/2*Cc2  2/Δt .* I ]

            A2 = [2/Δt .*M1[1] zero(M1[2]) zero(M1[1]);
                    zero(M1[1]) 2/Δt .* M1[2]  1/2*Cc'*M2;
                    zero(M1[1]) -1/2*Cc2  2/Δt .* I ];

            Bm1 = [1/Δt .* M1[1] zero(M1[2]) 1/2 .* M1[1] zero(M1[2]);
                    zero(M1[1])  1/Δt .* M1[2] zero(M1[2]) 1/2 .* M1[2];
                    -M1[1] * w_p^2/2 zero(M1[2]) 1/Δt*M1[1]  w_c/2*M_c;
                    zero(M1[1]) -M1[2]*w_p^2/2  -w_c/2*M_c' 1/Δt*M1[2]]

            Bm2 = [1/Δt .* M1[1] zero(M1[2]) -1/2 .* M1[1] zero(M1[2]);
                    zero(M1[1])  1/Δt .* M1[2] zero(M1[2]) -1/2 .* M1[2];
                    M1[1] * w_p^2/2 zero(M1[2]) 1/Δt*M1[1]  -w_c/2*M_c;
                    zero(M1[1]) M1[2]*w_p^2/2  w_c/2*M_c' 1/Δt*M1[2]];
            
            A1inv = inv(A1)
            Bm1inv = inv(Bm1);
            
            Eh1 = Eh[1]
            Eh2 = Eh[2]
            Jh1 = Jh[1]
            Jh2 = Jh[2]
            Bh = Bh

            U1 = [Eh1; Eh2; Bh] 
            U2 = [Eh1; Eh2; Jh1; Jh2]
            for i in 1:it
                    U1 = [Eh1; Eh2; Bh] 
                    U1 = A1inv*(A2 * U1)

                    Eh1 = U1[1:length(Eh1)]
                    Eh2 = U1[length(Eh1)+1:length(Eh1)+length(Eh2)]
                    Bh = U1[length(Eh1)+length(Eh2)+1:end]

                    U2 = [Eh1; Eh2; Jh1; Jh2]
                    U2 = Bm1inv*(Bm2 * U2)

                    Eh1 = U2[1:length(Eh1)]
                    Eh2 = U2[length(Eh1)+1:length(Eh1)+length(Eh2)]
                    Jh1 = U2[length(Eh1)+length(Eh2)+1:length(Eh1)+length(Eh2)+length(Jh1)]
                    Jh2 = U2[length(Eh1)+length(Eh2)+length(Jh1)+1:end]

                    U1 = [Eh1; Eh2; Bh] 
                    U1 = A1inv*(A2 * U1)

                    Eh1 = U1[1:length(Eh1)]
                    Eh2 = U1[length(Eh1)+1:length(Eh1)+length(Eh2)]
                    Bh = U1[length(Eh1)+length(Eh2)+1:end]

                    t += Δt
                    push!(En_E, (2π/k1)/(2*X.B.N*el[1]) * 2π/(2*Y.B.N*el[2]) * sqrt.(Eh1'*M1[1]*Eh1 + Eh2'*M1[2]*Eh2))
                    push!(En_J, (2π/k1)/(2*X.B.N*el[1]) * 2π/(2*Y.B.N*el[2]) * sqrt.(Jh1'*M1[1]*Jh1 + Jh2'*M1[2]*Jh2))
                    push!(En_B, (2π/k1)/(2*X.B.N*el[1]) * 2π/(2*Y.B.N*el[2]) * sqrt.(Bh'*M2*Bh))
                    push!(En, 1/2*En_E[end]^2 + 1/2*En_J[end]^2 + 1/2*En_B[end]^2 )

                    #push!(Enex_E, sqrt(hcubature(x -> dot(E(tt, x[1], x[2]), E(tt, x[1], x[2])), (0, 0), (2π/k1,2π))[1]))
                    #push!(Enex_J, sqrt(hcubature(x -> dot(J(tt, x[1], x[2]), J(tt, x[1], x[2])), (0, 0), (2π/k1,2π))[1]))
                    #push!(Enex_B, sqrt(hcubature(x -> dot(B(tt, x[1], x[2]), B(tt, x[1], x[2])), (0, 0), (2π/k1,2π))[1]))
                    #push!(Enex,  1/2*Enex_E[end]^2 + 1/2*Enex_B[end]^2 + 1/2*Enex_J[end]^2  )
            end

            Eh = SVector(Eh1, Eh2)
            Jh = SVector(Jh1, Jh2)
            Bh = Bh;
            
            t_n = t
            val_B = (i -> B.(t_n, xy2[i][1], xy2[i][2])).(1:length(xy2))
            val_E1 = (i -> E.(t_n, xy1_1[i][1], xy1_1[i][2])[1]).(1:length(xy1_1))
            val_E2 = (i -> E.(t_n, xy1_2[i][1], xy1_2[i][2])[2]).(1:length(xy1_2))
            val_J1 = (i -> J.(t_n, xy1_1[i][1], xy1_1[i][2])[1]).(1:length(xy1_1))
            val_J2 = (i -> J.(t_n, xy1_2[i][1], xy1_2[i][2])[2]).(1:length(xy1_2))
            
            En_mp_per = En 
            err_mp_E = max(maximum(abs.(colmat1_1 * Eh[1] - val_E1)), maximum(abs.(colmat1_2 * Eh[2] - val_E2)))
            err_mp_J = max(maximum(abs.(colmat1_1 * Jh[1] - val_J1)), maximum(abs.(colmat1_2 * Jh[2] - val_J2)))
            err_mp_B = maximum(abs.(colmat2 * Bh - val_B))

            #Cong Multipatch
            X,Y = SplineFEEC.PeriodicMultipatchTensorProductBSpline(domain, k, N, el).B
            V0 = X ⊗ Y
            V1 = SVector(X' ⊗ Y, X ⊗ Y')
            V2 = X' ⊗ Y'

            M1 = mass_matrix_d.(V1)
            M1inv = SVector(1/(Δy)^2, 1/(Δx)^2) .* inv.(M1)
            M1 = SVector((Δy)^2, (Δx)^2) .* M1
            M2 = mass_matrix_d(V2)

            #C = SplineFEEC.disc_diff(curl, V0)

            P = SplineFEEC.conforming_averaging_projection_matrix(V0)
            Px = SplineFEEC.conforming_averaging_projection_matrix_x(V0)
            Py = SplineFEEC.conforming_averaging_projection_matrix_y(V0)

            Cc = SplineFEEC.disc_diff(curl, V0).D[1] * Px
            Cc2 = disc_diff(curl, V0).D[1] * Px

            Eh = SVector(1/Δy, 1/Δx) .* L2_proj_d(E0, V1) 
            Jh = SVector(1/Δy, 1/Δx) .* L2_proj_d(J0, V1) 
            #SVector( M1_1\SplineFEEC.L2_prod_mixed_x((x,y) -> E0(x,y)[1], X' ⊗ Y),
            #                  M1_2\SplineFEEC.L2_prod_mixed_y((x,y) -> E0(x,y)[2], X ⊗ Y'))
            #Jh = SVector( M1_1\SplineFEEC.L2_prod_mixed_x((x,y) -> J0(x,y)[1], X' ⊗ Y),
            #                  M1_2\SplineFEEC.L2_prod_mixed_y((x,y) -> J0(x,y)[2], X ⊗ Y'))
            Bh = SplineFEEC.L2_proj_d(B0, V2)


            xy0, colmat0 = col_mat(V0, 10)
            xy1_1, colmat1_1 = SplineFEEC.col_mat_mixed_x(V1[1], 10) 
            xy1_2, colmat1_2 = SplineFEEC.col_mat_mixed_y(V1[2], 10) 
            xy2, colmat2 = col_mat_d(V2, 10)

            t = 0;
            En_E = []
            En_B = []
            En_J = []
            En = []
            #Enex_E = []
            #Enex_J = []
            #Enex_B = []
            #Enex = [];

            M_p = w_p^2 .* M1
            M_c = w_c .* SplineFEEC.mixed_mass_matrix_x_d(V1[1], V1[2]);
            
            A1 = [2/Δt .* M1[1]  zero(M1[2]) zero(Cc'*M2);
                    zero(M1[1]) 2/Δt .* M1[2]  -1/2*Cc'*M2;
                    zero(Cc2)  1/2*Cc2  2/Δt .* I ]

            A2 = [2/Δt .*M1[1] zero(M1[2]) zero(Cc'*M2);
                    zero(M1[1]) 2/Δt .* M1[2]  1/2*Cc'*M2;
                    zero(Cc2) -1/2*Cc2  2/Δt .* I ];

            Bm1 = [1/Δt .* M1[1] zero(M1[2]) 1/2 .* M1[1] zero(M1[2]);
                    zero(M1[1])  1/Δt .* M1[2] zero(M1[1]) 1/2 .* M1[2];
                    -M1[1] * w_p^2/2 zero(M1[2]) 1/Δt*M1[1]  w_c/2*M_c;
                    zero(M1[1]) -M1[2]*w_p^2/2  -w_c/2*M_c' 1/Δt*M1[2]]

            Bm2 = [1/Δt .* M1[1] zero(M1[2]) -1/2 .* M1[1] zero(M1[2]);
                    zero(M1[1])  1/Δt .* M1[2] zero(M1[1]) -1/2 .* M1[2];
                    M1[1] * w_p^2/2 zero(M1[2]) 1/Δt*M1[1]  -w_c/2*M_c;
                    zero(M1[1]) M1[2]*w_p^2/2  w_c/2*M_c' 1/Δt*M1[2]];
            
            A1inv = inv(A1)
            Bm1inv = inv(Bm1);
            
            Eh1 = Eh[1]
            Eh2 = Eh[2]
            Jh1 = Jh[1]
            Jh2 = Jh[2]
            Bh = Bh

            U1 = [Eh1; Eh2; Bh] 
            U2 = [Eh1; Eh2; Jh1; Jh2]
            for i in 1:it
                    U1 = [Eh1; Eh2; Bh] 
                    U1 = A1inv*(A2 * U1)

                    Eh1 = U1[1:length(Eh1)]
                    Eh2 = U1[length(Eh1)+1:length(Eh1)+length(Eh2)]
                    Bh = U1[length(Eh1)+length(Eh2)+1:end]

                    U2 = [Eh1; Eh2; Jh1; Jh2]
                    U2 = Bm1inv*(Bm2 * U2)

                    Eh1 = U2[1:length(Eh1)]
                    Eh2 = U2[length(Eh1)+1:length(Eh1)+length(Eh2)]
                    Jh1 = U2[length(Eh1)+length(Eh2)+1:length(Eh1)+length(Eh2)+length(Jh1)]
                    Jh2 = U2[length(Eh1)+length(Eh2)+length(Jh1)+1:end]

                    U1 = [Eh1; Eh2; Bh] 
                    U1 = A1inv*(A2 * U1)

                    Eh1 = U1[1:length(Eh1)]
                    Eh2 = U1[length(Eh1)+1:length(Eh1)+length(Eh2)]
                    Bh = U1[length(Eh1)+length(Eh2)+1:end]

                    t += Δt
                    push!(En_E, (2π/k1)/(2*X.B.N*el[1]) * 2π/(2*Y.B.N*el[2]) * sqrt.(Eh1'*M1[1]*Eh1 + Eh2'*M1[2]*Eh2))
                    push!(En_J, (2π/k1)/(2*X.B.N*el[1]) * 2π/(2*Y.B.N*el[2]) * sqrt.(Jh1'*M1[1]*Jh1 + Jh2'*M1[2]*Jh2))
                    push!(En_B, (2π/k1)/(2*X.B.N*el[1]) * 2π/(2*Y.B.N*el[2]) * sqrt.(Bh'*M2*Bh))
                    push!(En, 1/2*En_E[end]^2 + 1/2*En_J[end]^2 + 1/2*En_B[end]^2 )

                   # push!(Enex_E, sqrt(hcubature(x -> dot(E(tt, x[1], x[2]), E(tt, x[1], x[2])), (0, 0), (2π/k1,2π))[1]))
                   # push!(Enex_J, sqrt(hcubature(x -> dot(J(tt, x[1], x[2]), J(tt, x[1], x[2])), (0, 0), (2π/k1,2π))[1]))
                   # push!(Enex_B, sqrt(hcubature(x -> dot(B(tt, x[1], x[2]), B(tt, x[1], x[2])), (0, 0), (2π/k1,2π))[1]))
                   # push!(Enex,  1/2*Enex_E[end]^2 + 1/2*Enex_B[end]^2 + 1/2*Enex_J[end]^2  )
            end

            Eh = SVector(Eh1, Eh2)
            Jh = SVector(Jh1, Jh2)
            Bh = Bh;
            
            t_n = t
            val_B = (i -> B.(t_n, xy2[i][1], xy2[i][2])).(1:length(xy2))
            val_E1 = (i -> E.(t_n, xy1_1[i][1], xy1_1[i][2])[1]).(1:length(xy1_1))
            val_E2 = (i -> E.(t_n, xy1_2[i][1], xy1_2[i][2])[2]).(1:length(xy1_2))
            val_J1 = (i -> J.(t_n, xy1_1[i][1], xy1_1[i][2])[1]).(1:length(xy1_1))
            val_J2 = (i -> J.(t_n, xy1_2[i][1], xy1_2[i][2])[2]).(1:length(xy1_2))

            En_conga_per = En
            err_conga_E = max(maximum(abs.(colmat1_1 * Py * Eh[1] - val_E1)), maximum(abs.(colmat1_2 * Px * Eh[2] - val_E2)))
            err_conga_J = max(maximum(abs.(colmat1_1 * Py * Jh[1] - val_J1)), maximum(abs.(colmat1_2 * Px * Jh[2] - val_J2)))
            err_conga_B = maximum(abs.(colmat2 * Bh - val_B))

    return it, En_per, En_mp_per, En_conga_per, Enex_per
end

