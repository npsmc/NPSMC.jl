using Distributed
using Plots

addprocs(8)

@everywhere begin

    using LinearAlgebra
    using StatsBase

    function stochastic( β=2, n=200)
        h = n^-(1/3)
        x = 0:h:10
        N = length(x)
        d = (-2/h^2 .- x) + 2sqrt(h*β) * randn(N)
        e = ones(N-1) / h^2
        eigvals(SymTridiagonal(d, e))[N]
    end

    t = 10000

end

βs = [1, 2, 4, 10, 20]

@info "Sequential version"
p1 = plot(title="1 process")
@time for β=βs
    z = fit(Histogram, [stochastic(β) for i = 1:t], -4:0.01:1).weights
    plot!(p1, midpoints(-4:0.01:1), z/sum(z)/0.01)
end

@info " Parallel version "
p2 = plot(title="$(nprocs()) processes")
@time for β=βs
    z = @distributed (+) for p=1:nprocs()
        fit(Histogram, [stochastic(β) for i = 1:t], -4:0.01:1).weights
    end
    plot!(p2, midpoints(-4:0.01:1), z/sum(z)/0.01)
end

# Combine plots
plot(p1, p2, layout=(2, 1), size=(700, 700)) |> display

rmprocs(deleteat!(procs(), 1))
