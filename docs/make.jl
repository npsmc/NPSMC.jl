push!(LOAD_PATH, "../src/")
ENV["GKSwstype"]="100"

using DifferentialEquations
using DelimitedFiles
using Documenter
using Literate
using Plots
using NPSMC

# generate examples
OUTPUT = joinpath(@__DIR__, "src", "generated")

# examples = filter(x -> occursin(r".jl", x), map(relpath, readdir(joinpath(@__DIR__, "..", "examples"))))

examples = String[]
push!(examples, "sinus_data.jl")
push!(examples, "catalog.jl")
push!(examples, "model_forecasting.jl")
push!(examples, "analog_forecasting.jl")
push!(examples, "lorenz63.jl")
push!(examples, "lorenz96.jl")
#push!(examples, "monte_carlo.jl")

example_pages = Any[]

for example in examples
    EXAMPLE = joinpath(@__DIR__, "..", "examples", example)
    @show page = string("generated/", example[1:end-3], ".md")

    open(`head -n1 $EXAMPLE`) do io
        title = join(readdlm(io)[3:end], " ")
        push!(example_pages, title => page)
    end

    Literate.markdown(EXAMPLE, OUTPUT)
    #Literate.notebook(EXAMPLE, OUTPUT)
    #Literate.script(EXAMPLE, OUTPUT)
end

@show example_pages

pages = Any[
    "Home"=>"index.md",
    "Catalog"=>"catalog.md",
    "Data Assimilation"=>"data_assimilation.md",
    "State Space"=>"state-space.md",
    "Models"=>"models.md",
    "Ensemble Kalman filters"=>"ensemble_kalman_filters.md",
    "Ensemble Kalman smoothers"=>"ensemble_kalman_smoothers.md",
    "Forecasting"=>"forecasting.md",
    "Particle filters"=>"particle_filters.md",
    "Time Series"=>"time-series.md",
    "Utilities"=>"utils.md",
    "Examples"=>example_pages,
    "Some ideas"=>"ideas.md",
]

@show pages

makedocs(
    modules = [NPSMC],
    sitename = "NPSMC.jl",
    doctest = true,
    authors = "Pierre Navaro",
    format = Documenter.HTML(),
    pages = pages,
)

deploydocs(
    deps = Deps.pip("mkdocs", "python-markdown-math"),
    repo = "github.com/npsmc/NPSMC.jl.git",
)
