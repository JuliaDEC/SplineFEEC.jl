# SplineFEEC.jl Documentation


The goal of this package is the discretization and solution of differential equations based on a commuting diagram. Initially this will be limited to cardinal BSplines on tensor product domains.


### How to use the package in its current state
Firstly, clone the repository/ download the files such that you have them locally. 

Next, after starting `julia` in the terminal or in a jupyter notebook, we change the local path to the package directory (Note that `;` opens the shell environment in julia) via:
```julia
julia> ;
shell> cd "/path/to/SplineFEEC"
```


Now, we can activate the package (Note that `]` opens the package environment in julia) via:
```julia
julia> ]
(v1.5) pkg> activate .
```

Lastly, we can import the package:
```julia
julia> using SplineFEEC
```

Every function explicitly exported in `SplineFEEC.jl` can be accessed directly, everything else needs the prefix 
```julia
julia> SplineFEEC.functionname()
```

### Example Notebook
For a quick start, there is a notebook that can be found [here](https://home.in.tum.de/schnack/public/SplineFEEC-examples.ipynb).
The notebook contains all basic functionality and will continuously be extended, albeit it can also be found in more detail in the documentation soon. 