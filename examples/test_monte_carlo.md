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
using DifferentialEquations, Distributions
```

```julia
using NPSMC
```

```julia
x0 = [8.0;0.0;30.0]
tspan = (0.0,5.0)
p = [10.0,28.0,8/3]
prob = ODEProblem(lorenz63, x0, tspan, p)
x0 = last(solve(prob, reltol=1e-6, save_everystep=false))'
x0
```

```julia
using LinearAlgebra
np = 10
μ = [0.,0.,0.]
σ = 1.0 .* Matrix(I, 3, 3)
d = MvNormal( μ, σ)
x = x0 .+ rand(d, np)'
```

```julia
dt = 1.0
tspan = (0.0, dt)
x0    = [8.0;0.0;30.0]

prob  = ODEProblem( lorenz63, x0, tspan, p)

function prob_func( prob, i, repeat)
    prob.u0 .= x[i,:]
    prob
end

monte_prob = MonteCarloProblem(prob, prob_func=prob_func)

sim = solve(monte_prob, Tsit5(), num_monte=np)
```


```julia
using Plots

plot(sim)
```

```julia
xf = [last(sim[i].u) for i in 1:np]
```

```julia

```
