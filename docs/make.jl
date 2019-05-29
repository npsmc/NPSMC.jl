using Documenter, NPSMC

sources = filter(x -> occursin(r".md", x), map(relpath, 
                 readdir(joinpath(@__DIR__, "..", "examples"))))
examples = String[]
for source in sources
    @show source
    cp( joinpath(@__DIR__, "..",  "examples",  source),
        joinpath(@__DIR__, "src", "examples",  source))

    push!(examples, joinpath("examples",  source))
end

makedocs(
    modules   = [NPSMC],
    sitename  = "NPSMC.jl",
    doctest   = false, 
    authors   = "Pierre Navaro",
    format    = Documenter.HTML(),
    pages     = ["Home" => "index.md",
                 "Examples" => examples]
)

deploydocs(
    deps   = Deps.pip("mkdocs", "python-markdown-math"),
    repo   = "github.com/npsmc/NPSMC.jl.git",
 )
