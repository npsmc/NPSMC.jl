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

include("../src/models.jl")
include("../src/generate_data.jl")
include("../src/analog_forecasting.jl")


# +
σ = 10.0
ρ = 28.0
β = 8.0/3

dt_integration = 0.01
dt_states      = 1 
dt_obs         = 8 
parameters     = [σ, ρ, β]
var_obs        = [1]
nb_loop_train  = 10^2 
nb_loop_test   = 10
sigma2_catalog = 0.0
sigma2_obs     = 2.0

ssm = StateSpace( dt_integration, dt_states, dt_obs, 
                  parameters, var_obs,
                  nb_loop_train, nb_loop_test,
                  sigma2_catalog, sigma2_obs )

xt, yo, catalog = generate_data( ssm )
# -


af = AnalogForecasting( 5, xt, catalog )

yo = rand(10,3)

yo[yo.>0.5] .= NaN


yo

for k in 1:10
    ivar = findall(x -> x > 0.1, yo[k,:])
    @show ivar
end


