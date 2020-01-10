# # Analog forecasting

using Plots, DifferentialEquations, NPSMC


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

# # compute u0 to be in the attractor space

u0    = [8.0;0.0;30.0]
tspan = (0.0,5.0)
prob  = ODEProblem(ssm.model, u0, tspan, ssm.params)
u0    = last(solve(prob, reltol=1e-6, save_everystep=false))

# # Generate the data

xt, yo, catalog = generate_data( ssm, u0 )

# # Create the forecasting function

af = AnalogForecasting( 50, xt, catalog; 
    regression = :local_linear, sampling = :multinomial )

# # Data assimilation

np = 100
DA = DataAssimilation( af, xt, ssm.sigma2_obs)
x̂ = forecast(DA, yo, EnKS(np), progress = false);
println(RMSE(xt, x̂))

# # Plot results

plot(xt.t, xt[1], label=:true)
plot!(xt.t, x̂[1], label=:forecasted)
scatter!(yo.t, yo[1], markersize=2, label=:observed)
