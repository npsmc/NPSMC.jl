import Plots:plot, plot3d

function plot( x :: TimeSeries; kwargs... )

    p = plot()
    for i in 1:xt.nv
        plot!(p, xt.time,vcat(xt.values'...)[:,i], 
        line=(:solid,i), label="u$i")
    end
    p

end

function plot3d( x :: TimeSeries; kwargs... )

    p = plot3d(1, xlim=(-25,25), ylim=(-25,25), zlim=(0,50),
                title = "Lorenz 63", marker = 2)
    for x in eachrow(vcat(x̂.values'...))
        push!(p, x...)
    end
    p

end
