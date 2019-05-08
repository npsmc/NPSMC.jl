using Documenter
using NPSMC

makedocs(
    sitename = "NPSMC",
    format = Documenter.HTML(),
    modules = [NPSMC]
)

deploydocs(
    deps   = Deps.pip("mkdocs", "python-markdown-math"),
    repo   = "github.com/npsmc/NPSMC.jl.git",
 )
