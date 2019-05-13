# -*- coding: utf-8 -*-
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

include("src/models.jl")
include("src/generate_data.jl")

# +
σ = 10.0
ρ = 28.0
β = 8.0/3

dt_integration = 0.01
dt_states      = 1 
dt_obs         = 8 
parameters     = [σ, ρ, β]
var_obs        = [1]
nb_loop_train  = 10
nb_loop_test   = 1
sigma2_catalog = 0.0
sigma2_obs     = 2.0

ssm = StateSpace( dt_integration, dt_states, dt_obs, 
                  parameters, var_obs,
                  nb_loop_train, nb_loop_test,
                  sigma2_catalog, sigma2_obs )

xt, yo, catalog = generate_data( ssm )


# +
using Plots

plot(catalog.analogs)
plot!(catalog.successors)
# -


