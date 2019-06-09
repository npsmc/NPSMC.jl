---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.1.3
  kernelspec:
    display_name: Julia 1.1.1
    language: julia
    name: julia-1.1
---

```julia
using Plots
using NPSMC
using DifferentialEquations
```

```julia
?StateSpaceModel
```

```julia
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
```
```julia
plot( xt.t, vcat(xt.u'...)[:,1])
scatter!( yo.t, vcat(yo.u'...)[:,1]; markersize=2)
```

```julia
np = 100
data_assimilation = DataAssimilation( ssm, xt)
@time x̂ = data_assimilation(yo, PF(np));
RMSE(xt, x̂)
```

```julia
plot(xt.t, vcat(x̂.u'...)[:,1])
scatter!(xt.t, vcat(xt.u'...)[:,1]; markersize=2)
plot!(xt.t, vcat(x̂.u'...)[:,2])
scatter!(xt.t, vcat(xt.u'...)[:,2]; markersize=2)
plot!(xt.t, vcat(x̂.u'...)[:,3])
scatter!(xt.t, vcat(xt.u'...)[:,3]; markersize=2)
```

```julia
p = plot3d(1, xlim=(-25,25), ylim=(-25,25), zlim=(0,50),
            title = "Lorenz 63", marker = 2)
for x in eachrow(vcat(x̂.u'...))
    push!(p, x...)
end
p
```
