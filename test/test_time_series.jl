@testset " TimeSeries " begin

using Random

nt, nv = 10, 3
xt = TimeSeries{Float64}(nt, nv)

@test length(xt.time) == nt
@test typeof(xt.time) == Array{Float64,1}


time = collect(0:10.0)
values = rand( nv, nt)
yo = TimeSeries(time, values)

@test typeof(yo.time)   == Array{Float64,1}
@test typeof(yo.values) == Array{Float64,2}

end
