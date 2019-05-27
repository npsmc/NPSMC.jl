using Documenter, NPSMC

makedocs(
    modules   = [NPSMC],
    sitename  = "NPSMC.jl",
    doctest   = false, 
    authors   = "Pierre Navaro",
    format    = Documenter.HTML(),
    pages     = ["Home" => "index.md"]
)

deploydocs(
    deps   = Deps.pip("mkdocs", "python-markdown-math"),
    repo   = "github.com/npsmc/NPSMC.jl.git",
 )
