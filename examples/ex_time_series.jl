# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Julia 1.1.0
#     language: julia
#     name: julia-1.1
# ---

include("../src/models.jl")
include("../src/generate_data.jl")

nt, nv = 10, 3
xt = TimeSeries{Float64}(nt, nv)

using Random
time = collect(0:10.0)
values = rand( nt, nv)
yo = TimeSeries(time, values)

typeof(yo.time), typeof(yo.values)


