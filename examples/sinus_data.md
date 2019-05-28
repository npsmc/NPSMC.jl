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

```julia
using Plots, NPSMC
```

```julia
α = 3.0

dt_integration = 0.01
dt_states      = 1 
dt_obs         = 8 
params         = [α]
var_obs        = [1]
nb_loop_train  = 10^2 
nb_loop_test   = 10
sigma2_catalog = 0.0
sigma2_obs     = 0.1

ssm = StateSpaceModel( sinus, 
                       dt_integration, 
                       dt_states, 
                       dt_obs, 
                       params, 
                       var_obs,
                       nb_loop_train, 
                       nb_loop_test,
                       sigma2_catalog, 
                       sigma2_obs )

xt, yo, catalog = generate_data( ssm, [0.0] );

```

```julia
plot(xt.time, vcat(xt.values...)[:,1])
scatter!(yo.time, vcat(yo.values...)[:,1]; markersize=2)
```

```julia
scatter(catalog.analogs[1,:], catalog.successors[1,:], markersize=1)
```

```julia

```

```julia

```

```julia

```
