# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.2
#   kernelspec:
#     display_name: Julia 1.1.1
#     language: julia
#     name: julia-1.1
# ---

using Plots

include("../src/models.jl")
include("../src/time_series.jl")
include("../src/state_space.jl")
include("../src/catalog.jl")
include("../src/generate_data.jl")

σ = 10.0
ρ = 28.0
β = 8.0/3


# +
dt_integration = 0.01
dt_states      = 1 
dt_obs         = 8 
parameters     = [σ, ρ, β]
var_obs        = [1]
nb_loop_train  = 10
nb_loop_test   = 1
sigma2_catalog = 0.0
sigma2_obs     = 2.0

ssm = StateSpaceModel( dt_integration, dt_states, dt_obs, 
                       parameters, var_obs,
                       nb_loop_train, nb_loop_test,
                       sigma2_catalog, sigma2_obs )
# -


xt, yo, catalog = generate_data( ssm )


plot(catalog.analogs[1,:])
plot!(catalog.analogs[2,:])
plot!(catalog.analogs[3,:])

catalog.analogs[:,1]

p = plot3d(1, xlim=(-25,25), ylim=(-25,25), zlim=(0,50),
                title = "Lorenz 63", marker = 1)
for x in eachcol(catalog.analogs)
    push!(p, x...)
end
p


