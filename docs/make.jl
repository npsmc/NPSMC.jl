using DelimitedFiles
using Documenter
using Literate
using NPSMC
using Plots 

# generate examples
OUTPUT = joinpath(@__DIR__, "src", "generated")

examples = filter(x -> occursin(r".jl", x), map(relpath, readdir(joinpath(@__DIR__, "src", "examples"))))
example_pages = Any[]

for example in examples


    @show EXAMPLE = joinpath(@__DIR__, "src", "examples", example)
    @show page = string("generated/", example[1:end-3],".md")

    open(`head -n1 $EXAMPLE`) do io
         title = join(readdlm(io)[3:end], " ")
         push!(example_pages, title => page)
    end

    Literate.markdown(EXAMPLE, OUTPUT)
    #Literate.notebook(EXAMPLE, OUTPUT)
    #Literate.script(EXAMPLE, OUTPUT)
end

pages = Any["Home" => "index.md",
            "Models" => "models.md",
            "Data Assimilation" => "data_assimilation.md",
            "State Space" => "state_space.md",
            "Examples" => example_pages ]

makedocs(
    modules   = [NPSMC],
    sitename  = "NPSMC.jl",
    doctest   = true, 
    authors   = "Pierre Navaro",
    format    = Documenter.HTML(),
    pages     = pages
)

deploydocs(
    deps   = Deps.pip("mkdocs", "python-markdown-math"),
    repo   = "github.com/npsmc/NPSMC.jl.git",
 )
