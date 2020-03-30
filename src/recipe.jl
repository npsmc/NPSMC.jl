using Plots
gr()

struct TimeSeries
    t :: Vector{Float64}
    u :: Vector{Vector{Float64}}
end

tspan = 0:0.01:2π

ts = TimeSeries( tspan, [ [sin(θ), cos(θ), θ] for θ in tspan])

@recipe function f(ts::TimeSeries, var :: Int) 

    xlabel := :time
    ts.t, getindex.(ts.u, var)

end

plot(ts, 1, label =:sin)
plot!(ts, 2, label=:cos)
