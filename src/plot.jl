#=
import Plots: plot, plot3d

function plot(x::TimeSeries; kwargs...)

    p = plot()
    for i = 1:x.nv
        plot!(p, x.t, vcat(x.u'...)[:, i], line = (:solid, i), label = "u$i")
    end
    p

end

function plot3d(x::TimeSeries; kwargs...)

    p = plot3d(
        1,
        xlim = (-25, 25),
        ylim = (-25, 25),
        zlim = (0, 50),
        title = "Lorenz 63",
        marker = 2,
    )
    for x in eachrow(vcat(x̂.u'...))
        push!(p, x...)
    end
    p

end
=#
