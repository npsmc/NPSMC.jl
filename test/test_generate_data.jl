# -*- coding: utf-8 -*-
using DifferentialEquations, Distributions, Plots
using Random, LinearAlgebra

include("../src/models.jl")

include("../src/generate_data.jl")

# +
#@testset " Generate data for Lorenz-63 model " begin


    Random.seed!(1)

    σ = 10.0
    ρ = 28.0
    β = 8.0/3

    dt_integration = 0.01
    dt_states      = 1 
    dt_obs         = 8 
    params         = [σ, ρ, β]
    var_obs        = [1]
    nb_loop_train  = 10^2 
    nb_loop_test   = 10
    sigma2_catalog = 0.0
    sigma2_obs     = 2.0

    ssm = StateSpace( dt_integration, dt_states, dt_obs, 
                      params, var_obs,
                      nb_loop_train, nb_loop_test,
                      sigma2_catalog, sigma2_obs )

# +
@assert ssm.dt_states < ssm.dt_obs
# @error " ssm.dt_obs must be bigger than ssm.dt_states"
@assert mod(ssm.dt_obs,ssm.dt_states) == 0.0
# @error " ssm.dt_obs must be a multiple of ssm.dt_states "
# 5 time steps (to be in the attractor space)       
x0 = [8.0;0.0;30.0]
tspan = (0.0,5.0)
p = [10.0,28.0,8/3]
prob = ODEProblem(lorenz63, x0, tspan, p)

x0 = last(solve(prob, reltol=1e-6, save_everystep=false))
tspan = (0.0,ssm.nb_loop_test)
prob = ODEProblem(lorenz63, x0, tspan, p)

# generSate true state (xt)
sol = solve(prob,reltol=1e-6,saveat=dt_integration)
xt  = TimeSeries(sol.t, vcat(sol.u'...))

# generate  partial/noisy observations (yo)
d   = MvNormal(ssm.sigma2_obs .* Matrix(I,3,3))

yo     = TimeSeries( xt.time, xt.values .* NaN)
step = ssm.dt_obs ÷ ssm.dt_states
for j in ssm.var_obs
    for i in 1:step:nvalues
        yo.values[i,j] = xt.values[i,j] + eps[j,i]
    end
end
nvalues = length(xt.time)
eps = rand(d, nvalues)
plot(  xt.time, xt.values)
scatter!(yo.time, yo.values[:,var_obs])

# +
#generate catalog
x0 = last(sol)
tspan = (0.0,ssm.nb_loop_train)
prob = ODEProblem(lorenz63, x0, tspan, p)
sol = solve(prob,reltol=1e-6,saveat=dt_integration)
n = length(sol.t)
if ssm.sigma2_catalog > 0
    d   = MvNormal([0.,0.0,.0], ssm.sigma2_catalog .* Matrix(I,3,3))
    eta = rand(d, n)
    catalog_tmp = vcat(sol.u'...) .+ eta'
else
    catalog_tmp = vcat(sol.u'...) 
end

catalog = Catalog( catalog_tmp[1:end-dt_states,:],
                   catalog_tmp[dt_states:end,:], 
                   ssm)

   # @test true

#end
# -


