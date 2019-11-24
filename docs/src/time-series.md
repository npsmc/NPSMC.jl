# Time Series 

```@autodocs
Modules = [NPSMC]
Pages   = ["time_series.jl"]
```

```@example 1
using NPSMC
using Random

# Initialize a times series with size

nt, nv = 10, 3
xt = TimeSeries(nt, nv)

# Initialize a times series with values

time = collect(0:10.0)
values = [rand(nv) for i = 1:nt]
yo = TimeSeries(time, values)

```
