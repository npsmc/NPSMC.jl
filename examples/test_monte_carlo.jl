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

using DifferentialEquations, Distributions

include("../src/models.jl")

x0 = [8.0;0.0;30.0]
tspan = (0.0,5.0)
p = [10.0,28.0,8/3]
prob = ODEProblem(lorenz63, x0, tspan, p)
x0 = last(solve(prob, reltol=1e-6, save_everystep=false))'
x0

using LinearAlgebra
np = 10
μ = [0.,0.,0.]
σ = 1.0 .* Matrix(I, 3, 3)
d = MvNormal( μ, σ)
x = x0 .+ rand(d, np)'
x[1,:]

rand(d, np)

# +
dt = 10.0
tspan = (0.0, dt)
x0    = [8.0;0.0;30.0]

prob  = ODEProblem( lorenz63, x0, tspan, p)

function prob_func( prob, i, repeat)
    prob.u0 .= x[i,:]
    prob
end

monte_prob = MonteCarloProblem(prob, prob_func=prob_func)

sim = solve(monte_prob, Tsit5(), num_monte=np, save_everystep=false)


# +
using Plots

plot(sim)
# -

xf = [last(sim[i].u) for i in 1:np]


