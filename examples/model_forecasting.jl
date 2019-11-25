# # Model Forecasting


# ## Set the State-Space model
using Plots
using NPSMC
using DifferentialEquations

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

# - compute u0 to be in the attractor space

u0    = [8.0;0.0;30.0]
tspan = (0.0,5.0)
prob  = ODEProblem(ssm.model, u0, tspan, parameters)
u0    = last(solve(prob, reltol=1e-6, save_everystep=false))

# ## Generate data

xt, yo, catalog = generate_data( ssm, u0 );

plot( xt.t, vcat(xt.u'...)[:,1])
scatter!( yo.t, vcat(yo.u'...)[:,1]; markersize=2)

# ## Data assimilation with model forecasting

np = 100
data_assimilation = DataAssimilation( ssm, xt)
@time x̂ = data_assimilation(yo, PF(np), progress = false);
println(RMSE(xt, x̂))

# ## Plot the times series

plot(xt.t, vcat(x̂.u'...)[:,1])
scatter!(xt.t, vcat(xt.u'...)[:,1]; markersize=2)
plot!(xt.t, vcat(x̂.u'...)[:,2])
scatter!(xt.t, vcat(xt.u'...)[:,2]; markersize=2)
plot!(xt.t, vcat(x̂.u'...)[:,3])
scatter!(xt.t, vcat(xt.u'...)[:,3]; markersize=2)


# ## Plot the phase-space plot

p = plot3d(1, xlim=(-25,25), ylim=(-25,25), zlim=(0,50),
            title = "Lorenz 63", marker = 2)
for x in eachrow(vcat(x̂.u'...))
    push!(p, x...)
end
p
