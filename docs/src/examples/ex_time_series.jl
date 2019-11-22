# # Time Series

using NPSMC
using Random

nt, nv = 10, 3
xt = TimeSeries(nt, nv)

time = collect(0:10.0)
values = [rand(nv) for i = 1:nt]
yo = TimeSeries(time, values)

println(typeof(yo.t), typeof(yo.u))
