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
xt.values[1]
# + {}
import Plots:plot

function plot( x :: TimeSeries; kwargs... )

    p = plot()
    for i in 1:xt.nv
        plot!(p, xt.time,vcat(xt.values'...)[:,i], line=(:solid,i), label="x$i")
    end
    p

end
# -

plot(xt)
scatter!( yo.time, vcat(yo.values'...)[:,1]; markersize=2)

# +
include("../src/model_forecasting.jl")

mf = ModelForecasting( ssm )

include("../src/utils.jl")
include("../src/data_assimilation.jl")
np = 5
da = DataAssimilation( mf, :EnKs, np, xt, ssm.sigma2_obs)
da.xb .= [ 7.25667463, 11.74405549, 16.96737996]
x̂ = data_assimilation(yo, da);

# -

plot(xt.time, vcat(x̂.values'...)[:,1])
plot!(xt.time, vcat(xt.values'...)[:,1])

plot!(xt.values)
RMSE(xt.values, x̂.values)

p = plot3d(1, xlim=(-25,25), ylim=(-25,25), zlim=(0,50),
                title = "Lorenz 63", marker = 2)
for x in eachrow(x̂.values)
    push!(p, x...)
end
p

xpart = [zeros(Float64,(3,10)) for i in 1:5]

# +
xf = reshape(1:30, 3, 10)
w  = ones(10)

vec(sum(xf .* w', dims=2))
# -

xpart[1] .= xf

rand(MvNormal(da.xb, da.B), np)



