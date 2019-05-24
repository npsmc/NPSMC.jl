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
#     display_name: Julia 1.1.0
#     language: julia
#     name: julia-1.1
# ---

using Plots

include("../src/models.jl")
include("../src/time_series.jl")
include("../src/catalog.jl")
include("../src/state_space.jl")
include("../src/generate_data.jl")

# +
α = 3.0

dt_integration = 0.01
dt_states      = 1 
dt_obs         = 8 
params         = [α]
var_obs        = [1]
nb_loop_train  = 10^2 
nb_loop_test   = 10
sigma2_catalog = 0.0
sigma2_obs     = 0.1

ssm = StateSpaceModel( sinus, 
                       dt_integration, 
                       dt_states, 
                       dt_obs, 
                       params, 
                       var_obs,
                       nb_loop_train, 
                       nb_loop_test,
                       sigma2_catalog, 
                       sigma2_obs )

xt, yo, catalog = generate_data( ssm, [0.0] )

# -

plot(xt.time, xt.values[1,:])
scatter!(yo.time, yo.values[1,:]; markersize=2)

sum(.!isnan.(yo.values[1,:]))




