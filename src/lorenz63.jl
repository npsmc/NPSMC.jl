include("models.jl")
include("time_series.jl")
include("state_space.jl")
include("catalog.jl")
include("plot.jl")
include("generate_data.jl")
include("utils.jl")
include("model_forecasting.jl")
include("regression.jl")
include("analog_forecasting.jl")
include("data_assimilation.jl")

import DifferentialEquations: ODEProblem, solve

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

xt, yo, catalog = generate_data( ssm, u0 )

regression = :local_linear
sampling   = :gaussian
f  = AnalogForecasting( 100, xt, catalog; 
                        regression = regression,
                        sampling   = sampling )
method = EnKS(200)
DA = DataAssimilation( f, xt, ssm.sigma2_obs )
@time x̂  = forecast( DA, yo, method)
rmse = RMSE(xt, x̂) 
println( " rmse = $rmse ")
