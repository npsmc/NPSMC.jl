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

using LinearAlgebra
using Distributions, LinearAlgebra
using DifferentialEquations

include("../src/models.jl")
include("../src/time_series.jl")
include("../src/state_space.jl")
include("../src/catalog.jl")
include("../src/generate_data.jl")

?StateSpaceModel

# +
σ = 10.0
ρ = 28.0
β = 8.0/3

dt_integration = 0.01
dt_states      = 1 
dt_obs         = 8 
parameters     = [σ, ρ, β]
var_obs        = [1]
nb_loop_train  = 30 
nb_loop_test   = 1
sigma2_catalog = 0.0
sigma2_obs     = 2.0

ssm = StateSpaceModel( dt_integration, dt_states, dt_obs, 
                       parameters, var_obs,
                       nb_loop_train, nb_loop_test,
                       sigma2_catalog, sigma2_obs )

xt, yo, catalog = generate_data( ssm )
# + {}
include("../src/model_forecasting.jl")

mf = ModelForecasting( ssm )
# -

μ = [0.,0.,0]
σ = 0.1 .* Matrix(I,3,3)
d = MvNormal(μ, σ)
np = 10
x = [8.0 0.0 30.0] .+ rand(d, np)'
mf(x)

xb = xt.values[1,:]
xf = xb' .+ rand(d, np)'


include("../src/data_assimilation.jl")
da = DataAssimilation( mf, :EnKs, np, xt, ssm.sigma2_obs)
data_assimilation(yo, da)

# -

findall(.!isnan.(yo.values[k,1]))



