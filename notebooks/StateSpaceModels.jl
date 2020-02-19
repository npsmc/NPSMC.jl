# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Julia 1.3.1
#     language: julia
#     name: julia-1.3
# ---

using StateSpaceModels, LinearAlgebra, Plots

# # example with missing data

nt = 100
t = LinRange(0.0, 6π, nt)
y = zeros(Float64, (nt, 2))
y[:, 1] .= sin.(t) + 0.1 * randn(nt)
y[:, 2] .= cos.(t) + 0.1 * randn(nt)
y[4:4:96, :] .= NaN
model = linear_trend(y)

# +
ss = statespace(model)

plot(
    [y y],
    legend = :topleft,
    label = "observations",
    lw = 2,
    layout = (2, 1),
    xticks = 0:10:77,
    color = "black",
    grid = false,
)
plot!(
    [ss.filter.a[:, 1] ss.smoother.alpha[:, 1]],
    label = ["filtered state" "smoothed state"],
    color = ["indianred" "green"],
    lw = 2,
)

# +
using StateSpaceModels, Distributions, LinearAlgebra, Plots, Random

# Fix seed
Random.seed!(1)

# Number of observations
n = 400
# State dimension (2d position + 2d speed)
m = 4
# Measurements dimension (2d noisy position)
p = 2
# Control dimension (2d acceleration)
q = 2

# Damping ratio
ρ = 0.05
# Time delta
Δ = 1.0

# State transition matrix
T = kron(Matrix{Float64}(I, p, p), [1.0 (1.0 - ρ * Δ / 2.0) * Δ; 0.0 (1.0 - ρ * Δ)])
# Input matrix
R = kron(Matrix{Float64}(I, p, p), [0.5 * Δ^2; Δ])
# Output (measurement) matrix
Z = kron(Matrix{Float64}(I, p, p), [1.0 0.0])

# Generate random actuators
Q = 0.5 * Matrix{Float64}(I, q, q)
η = MvNormal(Q)

# Generate random measurement noise
H = 2.0 * Matrix{Float64}(I, p, p)
ε = MvNormal(H)

# Simulate vehicle trajectory
α = zeros(n + 1, m)
y = zeros(n, p)
for t in 1:n
    y[t, :] = Z * α[t, :] + rand(ε)
    α[t + 1, :] = T * α[t, :] + R * rand(η)  
end
α = α[1:n, :]

# User defined model
model = StateSpaceModel(y, Z, T, R)
# Estimate vehicle speed and position
ss = statespace(model)

ss.model.H
ss.model.Q

anim = @animate for i in 1:n
    plot(y[1:i, 1], y[1:i, 2], label="Measured position", line=:scatter, lw=2, markeralpha=0.2, color=:black, title="Vehicle tracking")
    plot!(α[1:i, 1], α[1:i, 3], label="True position", lw=3, color=:indianred)
    plot!(ss.filter.a[2:i+1, 1], ss.filter.a[2:i+1, 3], label="Estimated position", lw=2, color=:forestgreen)
end

gif(anim, "vehicle_tracking.gif", fps = 15)
# -


