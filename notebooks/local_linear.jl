# # Generate data
#
#  Generate data from the following linear and Gaussian state-space model:
# \begin{align}
#   x(k) & = 0.95 x(k-1) + \eta(k)\\
#   y(k) & = x(k) + \epsilon(k)
# \end{align}
# with $\eta(k) \sim \mathcal{N}\left(0,Q^{true}=1\right)$ and $\epsilon(k) \sim \mathcal{N}\left(0,R^{true}=1\right)$. Use the function *random.normal* to generate the Gaussian noises. Then, plot the $x$ and $y$ time series.

using Random
using Plots
using Distributions
using LinearAlgebra
using DataAssim
using NearestNeighbors

# +
Random.seed!(42)

# generate true state and noisy observations
Q_true = 1.0
R_true = 1.0
dQ = Normal(0.0, Q_true)
dR = Normal(0.0, R_true)
K = 100
x_true = zeros(Float64,K)
y = zeros(Float64,K)
for k in 2:K
    x_true[k] = 0.95 * x_true[k-1] + rand(dQ)
    y[k] = x_true[k] + rand(dR)
end

# plot results
plot(x_true,linecolor=:red,linewidth=2, label="True state")
scatter!(y, linecolor=:black, markersize=2,label="Noisy observations")
ylims!(minimum(y)-1,maximum(y)+1)
title!("Simulated data from a linear Gaussian state-space model", fontsize=20)

# +
n = 1
gaussian = ModelMatrix(0.95 .* Matrix(I,n,n))
H = Matrix(I,n,n)
Q = Matrix(I,n,n)
nmax = 100;
no = 5:nmax;
Pi = Matrix(I,n,n)
xit = [0.0]
# true run
(M::ModelMatrix)(t,x) = M.M*x + cholesky(Q).U * randn(n,1)
xt, yt = FreeRun(gaussian, xit, Q, H, nmax, no);
# add perturbations to initial condition
xi = xit + cholesky(Pi).U * randn(n)

# add perturbations to obs
m = 1
R = Matrix(I,m, m)
yo = zeros(m,length(no))
for i in 1:length(no)
  yo[:,i] = yt[:,i] .+ cholesky(R).U * randn(m,1)
end
# -

plot(vec(xt) ,linecolor=:red,linewidth=2, label="True state")
scatter!(vec(yo), mc=:black, ms=3, label="observation")



# +
using LowRankApprox

k
ivar
ivar_neighboor


X = zeros(Float64,(nvn, k))
Y = zeros(Float64,(nv, k))
w = zeros(Float64, k)

nv = length(ivar)
nvn = length(ivar_neighboor)

res = zeros(Float64, (nv, k))
beta = zeros(Float64, (nv, nvn + 1))
Xm = zeros(Float64, (nvn, 1))
Xr = ones(Float64, (nvn + 1, k))
Cxx = zeros(Float64, (nvn + 1, nvn + 1))
Cxx2 = zeros(Float64, (nvn + 1, nvn + 1))
Cxy = zeros(Float64, (nv, nvn + 1))
pred = zeros(Float64, (nv, k))
X0r = ones(Float64, (nvn + 1, 1))

new(k, ivar, ivar_neighboor, res, beta, Xm, Xr, Cxx, Cxx2, Cxy, pred, X0r)
# -

function compute(ll::LocalLinear, x, xf_tmp, xf_mean, ip, X, Y, w)

    ivar = ll.ivar
    ivar_neighboor = ll.ivar_neighboor

    ll.Xm .= sum(X .* w', dims = 2)
    ll.Xr[2:end, :] .= X .- ll.Xm
    mul!(ll.Cxx, (ll.Xr .* w'), ll.Xr')
    mul!(ll.Cxx2, (ll.Xr .* w' .^ 2), ll.Xr')
    mul!(ll.Cxy, (Y .* w'), ll.Xr')

    if any(isnan.(ll.Cxx)) throw(" Some nan values in Cxx ") end

    U, S, V = psvd(ll.Cxx, rtol = 0.01)

    ll.Cxx .= V * diagm(1 ./ S) * U'
    ll.Cxx2 .= ll.Cxx2 * ll.Cxx
    # regression on principal components
    mul!(ll.beta, ll.Cxy, ll.Cxx)
    ll.X0r[2:end, :] .= x[ivar_neighboor, ip] .- ll.Xm
    # weighted mean
    xf_mean[ivar, ip] = ll.beta * ll.X0r
    mul!(ll.pred, ll.beta, ll.Xr)
    Y .-= ll.pred
    xf_tmp[ivar, :] .= xf_mean[ivar, ip] .+ Y
    # weigthed covariance
    cov_xf = (Y * (w .* Y')) ./ (1 .- tr(ll.Cxx2))
    cov_xf .= Symmetric(cov_xf .* (1 .+ tr(ll.Cxx2 * ll.X0r * ll.X0r' * ll.Cxx)))

    return cov_xf

end
