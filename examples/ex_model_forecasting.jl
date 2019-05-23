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
nb_loop_train  = 100 
nb_loop_test   = 10
sigma2_catalog = 0.0
sigma2_obs     = 2.0

ssm = StateSpaceModel( dt_integration, dt_states, dt_obs, 
                       parameters, var_obs,
                       nb_loop_train, nb_loop_test,
                       sigma2_catalog, sigma2_obs )

xt, yo, catalog = generate_data( ssm )
# -
# state and observations (when available)
plot(xt.time,xt.values[:,1], line=(:solid,:red), label="xt")
obs = findall(.!isnan.(yo.values))
scatter!(yo.time[obs],yo.values[obs],  label="yo")
plot!(xt.time,xt.values[:,2], line=(:solid,:blue), label="x₂")
#plot!(yo.time,yo.values[:,2], line=(:dot,:blue) )
plot!(xt.time,xt.values[:,3], line=(:solid,:green), label="x₃")
#plot!(yo.time,yo.values[:,3], line=(:dot,:green))
xlabel!("Lorenz-63 times")
title!("Lorenz-63 true (continuous lines) and observed trajectories (points)")

# +
include("../src/model_forecasting.jl")

mf = ModelForecasting( ssm )

# -

include("../src/utils.jl")
include("../src/data_assimilation.jl")
np = 100
da = DataAssimilation( mf, :EnKs, np, xt, ssm.sigma2_obs)
x̂ = data_assimilation(yo, da)

plot(x̂.values, linestyle=:dot)
plot!(xt.values)
RMSE(xt.values, x̂.values)

p = plot3d(1, xlim=(-25,25), ylim=(-25,25), zlim=(0,50),
                title = "Lorenz 63", marker = 2)
for x in eachrow(x̂.values)
    push!(p, x...)
end
p
