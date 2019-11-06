# -*- coding: utf-8 -*-
import Pkg; Pkg.add("DataAssim")

using DataAssim, Plots, Random, LinearAlgebra, Statistics
using DifferentialEquations

dt = 0.01
ℳ = Lorenz63Model(dt) # 

nv, nt = 3, 10000
x = randn(3, nt)
for k = 1:nt-1
    x[:,k+1] = ℳ(k,x[:,k])
end

p = plot3d(1, xlim=(-25,25), ylim=(-25,25), zlim=(0,50),
            title = "Lorenz 63", marker = 2)
for x in eachcol(x)
    push!(p, x...)
end
p

# Compute xit to be in the attractor space

# +
function lorenz63(du, u, p, t)

    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[2] - u[3]) - u[2]
    du[3] = u[1] *  u[2] - p[3]  * u[3]

end

x0    = [8.0;0.0;30.0]
tspan = (0.0,5.0)
σ     = 10.
β     = 8/3.
ρ     = 28.
p     = [σ, ρ, β]
prob  = ODEProblem(lorenz63, x0, tspan, p)
xit   = last(solve(prob, reltol=1e-6, save_everystep=false))

# +
n = 3
H = [1 0 0];
Q = Matrix(I,n,n)
nmax = 1000;
no = 5:nmax;
Pi = Matrix(3*I,n,n)

# true run
xt,yt = FreeRun(ℳ, xit, Q, H, nmax, no);

# add perturbations to IC
xi = xit + cholesky(Pi).U * randn(n)

# add perturbations to obs
m = 1
R = Matrix(I,m, m)
yo = zeros(m,length(no))
for i in 1:length(no)
  yo[:,i] = yt[:,i] .+ cholesky(R).U * randn(m,1)
end
# free run
xfree, yfree = FreeRun(ℳ, xi, Q, H, nmax, no)

# assimilation
xa, Pa = KalmanFilter(xi, Pi, ℳ, Q, yo, R, H, nmax, no)

rmse = sqrt(mean(((xt .- xa)).^2)) 
println( "rmse = $rmse")

plot(xt[1,:];    color=:blue,  label = "true")
plot!(xfree[1,:], color=:red,   label = "free")
plot!(xa[1,:],    linewidth=2, linecolor=:green,  label = "assim")
scatter!( yo[1,:]; label = :obs, markersize=1)
# -


