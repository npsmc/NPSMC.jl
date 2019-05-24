import Plots:plot

function plot( x :: TimeSeries; kwargs... )

    p = plot()
    for i in 1:xt.nv
        plot!(p, xt.time,vcat(xt.values'...)[:,i], line=(:solid,i), label="u$i")
    end
    p

end
