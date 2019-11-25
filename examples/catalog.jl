# # Example Catalog

using Plots, NPSMC

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


xt, yo, catalog = generate_data( ssm , [10.0;0.0;0.0]);


plot(catalog.analogs[1,:])
plot!(catalog.analogs[2,:])
plot!(catalog.analogs[3,:])


p = plot3d(1, xlim=(-25,25), ylim=(-25,25), zlim=(0,50),
                title = "Lorenz 63", marker = 1)
for x in eachcol(catalog.analogs)
    push!(p, x...)
end
p
