using SplineFEEC
using Documenter

makedocs(;
    modules=[SplineFEEC],
    authors="JuliaDEC",
    repo="https://github.com/FrederikSchnack/SplineFEEC.jl/blob/{commit}{path}#L{line}",
    sitename="SplineFEEC.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://FrederikSchnack.github.io/SplineFEEC.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/FrederikSchnack/SplineFEEC.jl",
)
