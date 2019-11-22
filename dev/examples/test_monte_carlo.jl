# # Monte-Carlo

using DifferentialEquations
using Distributions
using LinearAlgebra
using NPSMC
using Plots

x0 = [8.0;0.0;30.0]
tspan = (0.0,5.0)
p = [10.0,28.0,8/3]
prob = ODEProblem(lorenz63, x0, tspan, p)
x0 = last(solve(prob, reltol=1e-6, save_everystep=false))'

np = 10
μ = [0.,0.,0.]
σ = 1.0 .* Matrix(I, 3, 3)
d = MvNormal( μ, σ)
x = x0 .+ rand(d, np)'

dt = 1.0
tspan = (0.0, dt)
x0    = [8.0;0.0;30.0]

prob  = ODEProblem( lorenz63, x0, tspan, p)

function prob_func( prob, i, repeat)
    prob.u0 .= x[i,:]
    prob
end

monte_prob = MonteCarloProblem(prob, prob_func=prob_func)

sim = solve(monte_prob, Tsit5(), trajectories=np)

plot(sim)
