using SplineFEEC
using Documenter

makedocs(;
    modules=[SplineFEEC],
    authors="JuliaDEC",
    repo="https://github.com/JuliaDEC/SplineFEEC.jl/blob/{commit}{path}#L{line}",
    sitename="SplineFEEC.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JuliaDEC.github.io/SplineFEEC.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Basis" => "bases.md",
        "Operator" => "operator.md",
        "Examples" => "examples.md",
        "Library" => "library.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaDEC/SplineFEEC.jl",
    devbranch = "documentation",
)
