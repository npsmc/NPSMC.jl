# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Julia 1.2.0
#     language: julia
#     name: julia-1.2
# ---

using Pkg
Pkg.add(["EnKF", "Distributions", "ProgressMeter", "OrdinaryDiffEq", "LinearAlgebra", "DocStringExtensions"])

using EnKF
using Distributions
using DocStringExtensions
using LinearAlgebra
using ProgressMeter
using OrdinaryDiffEq
using Plots

# +
function lorenz(du,u,p,t)
 du[1] = 10.0*(u[2]-u[1])
 du[2] = u[1]*(28.0-u[3]) - u[2]
 du[3] = u[1]*u[2] - (8/3)*u[3]
end

u0 = [10.0; -5.0; 2.0]
tspan = (0.0,40.0)

Δt = 1e-2
T = tspan[1]:Δt:tspan[end]

prob = ODEProblem(lorenz,u0,tspan)
sol = solve(prob, RK4(), adaptive = false, dt = Δt)

integrator = init(prob, RK4(), adaptive =false, dt = Δt, save_everystep=false)
# -

plot(sol)

function (::PropagationFunction)(t::Float64, ENS::EnsembleState{N, TS}) where {N, TS}
    for (i,s) in enumerate(ENS.S)
        
        set_t!(integrator, deepcopy(t))
        set_u!(integrator, deepcopy(s))
        for j=1:50
        step!(integrator)
        end
        ENS.S[i] = deepcopy(integrator.u)

    end
    
    return ENS
end

fprop = PropagationFunction()

function (::MeasurementFunction)(t::Float64, s::TS) where TS
    return [s[1]+s[2]+s[3]]
end

function (::MeasurementFunction)(t::Float64) 
    return reshape([1.0, 1.0 , 1.0],(1,3))
end

m = MeasurementFunction()

function (::RealMeasurementFunction)(t::Float64, ENS::EnsembleState{N, TZ}) where {N, TZ}
    let s = sol(t)
    fill!(ENS, [deepcopy(s[1]+s[2]+s[3])])
    end
    return ENS
end

z = RealMeasurementFunction()

A = MultiAdditiveInflation(3, 1.05, MvNormal(zeros(3), 1.0*I))

ϵ = AdditiveInflation(MvNormal(zeros(1), 3.0*I))

N = 10
NZ = 1
isinflated = true
isfiltered = false
isaugmented = false

u0

# +
ens = initialize(N, MvNormal([20.0, -10.0, 10.0], 2.0*I))
estimation_state = [deepcopy(ens.S)]

tmp = deepcopy(u0)
true_state = [deepcopy(u0)]
# -

g = FilteringFunction()

enkf = ENKF(N, NZ, fprop, A, g, m, z, ϵ, isinflated, isfiltered, isaugmented)

# +
Δt = 1e-2
Tsub = 0.0:50*Δt:40.0-50*Δt

@showprogress for (n,t) in enumerate(Tsub)

    global ens
#     enkf.f(t, ens)
    t, ens,_ = enkf(t, 50*Δt, ens)
    push!(estimation_state, deepcopy(ens.S))
    

end

# +
s =  hcat(sol(T).u...)
ŝ =  hcat(mean.(estimation_state)...)

plt = plot(layout = (3, 1), legend = true)
plot!(plt[1], T, s[1,1:end], linewidth = 2, label = "truth")
scatter!(plt[1], Tsub, ŝ[1,1:end-1], linewidth = 2, markersize = 3, label = "EnKF mean", xlabel = "t", ylabel = "x", linestyle =:dash)

plot!(plt[2], T, s[2,1:end], linewidth = 2, label = "truth")
scatter!(plt[2], Tsub, ŝ[2,1:end-1], linewidth = 2, markersize = 3, label = "EnKF mean", xlabel = "t", ylabel = "y", linestyle =:dash)

plot!(plt[3], T, s[3,1:end], linewidth = 2, label = "truth")
scatter!(plt[3], Tsub, ŝ[3,1:end-1], linewidth = 2, markersize = 3, label = "EnKF mean", xlabel = "t", ylabel = "z", linestyle =:dash)

# -

plot(T, s[1,:], linewidth = 3, label = "truth")
# plot!(Tsub, ŝ[1,1:end-1], linewidth = 3, label = "EnKF mean", xlabel = "t", ylabel = "x", linestyle =:dash)
scatter!(Tsub, ŝ[1,1:end-1], linewidth = 3, label = "EnKF mean", xlabel = "t", ylabel = "x", linestyle =:dash)

plot(s[1,:], s[2,:], s[3,:], linewidth = 2, label = "truth", legend = true)
plot!(ŝ[1,1:end-1], ŝ[2,1:end-1], ŝ[3,1:end-1], linewidth = 2, label = "EnKF mean", xlabel = "x", 
    ylabel = "y", zlabel ="z", linestyle = :solid)
scatter!(ŝ[1,:], ŝ[2,:], ŝ[3,:], linewidth = 2, label = "EnKF mean", xlabel = "x", 
    ylabel = "y", zlabel ="z", linestyle = :solid)


