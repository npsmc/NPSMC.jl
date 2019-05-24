# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.3
#   kernelspec:
#     display_name: Julia 1.1.1
#     language: julia
#     name: julia-1.1
# ---

using Plots
using LinearAlgebra
using Distributions, LinearAlgebra
using DifferentialEquations

include("../src/models.jl")
include("../src/time_series.jl")
include("../src/state_space.jl")
include("../src/catalog.jl")


?StateSpaceModel

# +
include("../src/generate_data.jl")
σ = 10.0
ρ = 28.0
β = 8.0/3

dt_integration = 0.01
dt_states      = 1 
dt_obs         = 8 
parameters     = [σ, ρ, β]
var_obs        = [1]
nb_loop_train  = 100 
nb_loop_test   = 10
sigma2_catalog = 0.0
sigma2_obs     = 2.0

ssm = StateSpaceModel( lorenz63,
                       dt_integration, dt_states, dt_obs, 
                       parameters, var_obs,
                       nb_loop_train, nb_loop_test,
                       sigma2_catalog, sigma2_obs )

# compute u0 to be in the attractor space
u0    = [8.0;0.0;30.0]
tspan = (0.0,5.0)
prob  = ODEProblem(ssm.model, u0, tspan, parameters)
u0    = last(solve(prob, reltol=1e-6, save_everystep=false))

xt, yo, catalog = generate_data( ssm, u0 );
# -
include("../src/plot.jl")

plot(xt)
scatter!( yo.time, vcat(yo.values'...)[:,1]; markersize=2)

# +
include("../src/model_forecasting.jl")

mf = ModelForecasting( ssm )

include("../src/utils.jl")
include("../src/data_assimilation.jl")
np = 100
da = DataAssimilation( mf, :EnKs, np, xt, ssm.sigma2_obs)
@time x̂ = data_assimilation(yo, da);
RMSE(xt, x̂)
# -

plot(xt.time, vcat(x̂.values'...)[:,1])
scatter!(xt.time, vcat(xt.values'...)[:,1]; markersize=2)
plot!(xt.time, vcat(x̂.values'...)[:,2])
scatter!(xt.time, vcat(xt.values'...)[:,2]; markersize=2)
plot!(xt.time, vcat(x̂.values'...)[:,3])
scatter!(xt.time, vcat(xt.values'...)[:,3]; markersize=2)

p = plot3d(1, xlim=(-25,25), ylim=(-25,25), zlim=(0,50),
            title = "Lorenz 63", marker = 2)
for x in eachrow(vcat(x̂.values'...))
    push!(p, x...)
end
p
