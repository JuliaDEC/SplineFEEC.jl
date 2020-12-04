```@meta
CurrentModule = SplineFEEC
```

# SplineFEEC

```@index
```
### How to use the package in the current state
Firstly, clone the repository/ download the files sucht that you have them locally. 
Next, after starting `julia` in the terminal or the jupyter notebook, we change the local path to the package directory via
`;cd "/path/to/SplineFEEC"` (Note that `;` opens the shell environment in julia)
Now we can activate the package via `]activate .` (Note that `]` opens the package environment in julia)
Lastly, we can import the package via `using SplineFEEC`.

Every function explicitly exported in `SplineFEEC.jl` can be accessed directly, everything else needs the prefix `SplineFEEC.functionname()`. 

```@autodocs
Modules = [SplineFEEC]
```
