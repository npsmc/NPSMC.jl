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
#     display_name: Julia 1.1.0
#     language: julia
#     name: julia-1.1
# ---

using Plots, DifferentialEquations

include("../src/models.jl")
include("../src/time_series.jl")
include("../src/state_space.jl")
include("../src/catalog.jl")
include("../src/plot.jl")
include("../src/generate_data.jl")
include("../src/utils.jl")
include("../src/model_forecasting.jl")


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

xt, yo, catalog = generate_data( ssm, u0 );
typeof(xt)
# -
include("../src/analog_forecasting.jl")
include("../src/data_assimilation.jl")
af = AnalogForecasting( 50, xt, catalog; 
    regression = :local_linear, sampling = :multinomial )
np = 100
da = DataAssimilation( af, :EnKS, np, xt, ssm.sigma2_obs)
@time x̂ = data_assimilation(yo, da);
RMSE(xt, x̂)

plot(xt.t, vcat(xt.u'...)[:,1])
plot!(xt.t, vcat(x̂.u'...)[:,1])
scatter!(yo.t, vcat(yo.u'...)[:,1], markersize=2)


