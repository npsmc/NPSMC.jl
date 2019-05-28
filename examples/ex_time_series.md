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
using NPSMC
```

```julia
nt, nv = 10, 3
xt = TimeSeries{Float64}(nt, nv)
```

```julia
using Random
time = collect(0:10.0)
values = [rand(nv) for i = 1:nt]
yo = TimeSeries(time, values)
```

```julia
typeof(yo.time), typeof(yo.values)
```
