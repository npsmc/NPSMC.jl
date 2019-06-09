---
jupyter:
  jupytext:
    formats: ipynb,jl:light
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

# GENERATE SIMULATED DATA (LORENZ-96 MODEL)

```julia
using Plots, DifferentialEquations, Random
```

```julia
include("../src/models.jl")
include("../src/time_series.jl")
include("../src/state_space.jl")
include("../src/catalog.jl")
include("../src/generate_data.jl")
```

```julia
F = 8
J = 40 :: Int64
parameters = [F, J]
dt_integration = 0.05 # integration time
dt_states = 1 # number of integration times between consecutive states (for xt and catalog)
dt_obs = 4 # number of integration times between consecutive observations (for yo)
var_obs = randperm(MersenneTwister(1234), J)[1:20] # indices of the observed variables
nb_loop_train = 100 # size of the catalog
nb_loop_test = 10 # size of the true state and noisy observations
sigma2_catalog = 0.   # variance of the model error to generate the catalog   
sigma2_obs = 2. # variance of the observation error to generate observations

ssm = StateSpaceModel( lorenz96, 
                       dt_integration, dt_states, dt_obs, 
                       parameters, var_obs,
                       nb_loop_train, nb_loop_test,
                       sigma2_catalog, sigma2_obs )
```

5 time steps (to be in the attractor space)

Checks if the inegration function is well implemented.

```julia
u0 = F .* ones(Float64, J)
u0[J÷2] = u0[J÷2] + 0.01

tspan = (0.0,5.0)
p = [F, J]
prob  = ODEProblem(lorenz96, u0, tspan, p)
sol = solve(prob, reltol=1e-6, saveat= dt_integration)
x1  = [x[1] for x in sol.u]
x20 = [x[20] for x in sol.u]
x40 = [x[40] for x in sol.u]
plot(sol.t, x1)
plot!(sol.t, x20)
plot!(sol.t, x40)
```

Generate data and catalog

```julia
# run the data generation
xt, yo, catalog = generate_data(ssm, u0);
```

### PLOT STATE, OBSERVATIONS AND CATALOG

```julia
# state and observations (when available)
plot(xt.time,  vcat(xt.values'...)[:,1], line=(:solid,:red), label="x1")
scatter!(yo.time, vcat(yo.values'...)[:,1], markersize=2)
plot!(xt.time, vcat(xt.values'...)[:,20],line=(:solid,:blue), label="x20")
scatter!(yo.time, vcat(yo.values'...)[:,20],markersize=2)
plot!(xt.time, vcat(xt.values'...)[:,40],line=(:solid,:green), label="x40")
scatter!(yo.time, vcat(yo.values'...)[:,40],markersize=2)
xlabel!("Lorenz-96 times")
title!("Lorenz-96 true (continuous lines) and observed trajectories (dots)")
```

```julia

```
