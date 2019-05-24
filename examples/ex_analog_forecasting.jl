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
include("../src/generate_data.jl")

?StateSpaceModel

# +
σ = 10.0
ρ = 28.0
β = 8.0/3

dt_integration = 0.01
dt_states      = 1 
dt_obs         = 8 
var_obs        = [1]
nb_loop_train  = 100 
nb_loop_test   = 10
sigma2_catalog = 0.0
sigma2_obs     = 2.0

ssm = StateSpaceModel( lorenz63,
                       dt_integration, dt_states, dt_obs, 
                       [σ, ρ, β], var_obs,
                       nb_loop_train, nb_loop_test,
                       sigma2_catalog, sigma2_obs )

# compute u0 to be in the attractor space
u0    = [8.0;0.0;30.0]
tspan = (0.0,5.0)
prob  = ODEProblem(ssm.model, u0, tspan, ssm.params)
u0    = last(solve(prob, reltol=1e-6, save_everystep=false))

xt, yo, catalog = generate_data( ssm, u0 )
# + {}
# state and observations (when available)
plot(xt.time,xt.values[1,:], line=(:solid,:red), label="u1")

scatter!(yo.time, yo.values[1,:],  markersize=2, label="yo")
plot!(xt.time,xt.values[2,:], line=(:solid,:blue), label="u2")
plot!(xt.time,xt.values[3,:], line=(:solid,:green), label="u3")
xlabel!("Lorenz-63 times")
title!("Lorenz-63 true (continuous lines) and observed trajectories (points)")
# -

using NearestNeighbors
kdtree = KDTree( xt.values, leafsize=50)

# +
k = 50
# Multiple points
points = rand(3,100);

idxs, dists = knn(kdtree, points, k, true)
reshape(vcat(dists...),100,50)
# -

median(Iterators.flatten(dists))

# +
include("../src/model_forecasting.jl")
include("../src/analog_forecasting.jl")


mf = AnalogForecasting( 50, xt, catalog )
nt, nv, np = 1000, 3, 100
x = rand(np,nv)
mf( x )
# -

include("../src/utils.jl")
include("../src/data_assimilation.jl")
np = 100
da = DataAssimilation( mf, :EnKs, np, xt, ssm.sigma2_obs)
x̂ = data_assimilation(yo, da)

plot(x̂.values)

p = plot3d(1, xlim=(-25,25), ylim=(-25,25), zlim=(0,50),
                title = "Lorenz 63", marker = 2)
@gif for i=1:1000
    push!(p, x̂.values[i,:]...)
end


