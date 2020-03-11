# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.5
#   kernelspec:
#     display_name: Julia 1.3.1
#     language: julia
#     name: julia-1.3
# ---

# ### PROBLEM STATEMENT
#
# Data assimilation are numerical methods used in geophysics to mix the information of observations (noted as $y$) and a dynamical model (noted as $\mathcal{M}$). The goal is to estimate the true/hidden state of the system (noted as $x$) at every time step $k$. Usually, data assimilation problems are stated as nonlinear state-space models:
#
# \begin{align*}
# x(k) & = \mathcal{M}\left(x(k-1)\right) + \eta(k) \\
# y(k) & = \mathcal{H}\left(x(k)\right) + \epsilon(k)
# \end{align*}
#
# with $\eta$ and $\epsilon$ some independent white Gaussian noises respectively representing the model and the observation errors. These errors are supposed to be unbiased with covariances noted $Q$ and $R$. Here, we propose to use the Expectation-Maximization (EM) algorithm to estimate $Q$ and $R$.

# ### The batch EM algorithm for state-space models
#
# EM algorithm is a common algorithm in the statistical society. The goal of EM is to maximize a likelihood function. In our case (state-space models), the likelihood function to maximize is 
# \begin{align*}
# %& p\left(y(1:K), x(0:K)|Q, R \right) = & \nonumber \\ 
# & \mathcal{L}(Q,R) = p\left(x(0)\right) \prod_{k=1}^K p\left(x(k)|x(k-1),Q\right) \prod_{k=1}^K p\left(y(k)|x(k),R\right) & 
# \end{align*}
# where the first term corresponds to the initial condition, the second term to the dynamic equation, the third term to the observation equation. Note that $K$ is the total number of observations.
#
# The $\mathcal{L}(Q,R)$ likelihood is said to be "total" because we consider the all state-space model at all time steps. In practice, we can not maximize directly this likelihood w.r.t. $Q$ and $R$ because the sequence $x(0),\dots,x(K)$ is unknown and depends itself on $Q$ and $R$. We thus need an iterative procedure to first estimate the sequence $x(0),\dots,x(K)$ and then to update $Q$ and $R$. This is the purpose of the EM algorithm.
#
# The 2 main steps in EM are (1) the Expectation and (2) the Maximization. EM algorithm is iterative and we note $j$ the current iteration. Starting from an initial condition $Q^{(0)}$ and $R^{(0)}$, EM repeats these 2 steps until converge:
#
# 1) E-step: compute the "expectation" of the likelihood function conditionally on the previous estimates $Q^{(j-1)}$ and $R^{(j-1)}$. Mathematically, it is written as $E\left[\mathcal{L}(Q,R)|y(0),\dots,y(K),Q^{(j-1)},R^{(j-1)}\right]$. This expectation is also conditionally on the whole sequence of observations $y(0),\dots,y(K)$. We thus need to use the Kalman smoother to approximate as best this expectation.
#
# 2) M-step: "maximize" $E\left[\mathcal{L}(Q,R)|y(0),\dots,y(K),Q^{(j-1)},R^{(j-1)}\right]$ w.r.t. $Q$ and $R$. This maximization can be done using optimization techniques (e.g., gradient descent) or using an analytic formula. The resulting covariance estimates are $Q^{(j)}$ and $R^{(j)}$. 

# ### EM algorithm using the linear and Gaussian state-space model
#
# Here, we consider the case where $\mathcal{M}$ and $\mathcal{H}$ are linear operators (i.e. $M$ and $H$ matrices). The EM for the corresponding linear and Gaussian state-space model is known for a long time and given in Shumway and Stoffer (1982), see https://onlinelibrary.wiley.com/doi/epdf/10.1111/j.1467-9892.1982.tb00349.x. This algorithm is coded in the **pykalman** Python library. Here, we aim to apply the EM algorithm on a simple problem.

# **Question)** Generate data from the following linear and Gaussian state-space model:
# \begin{align}
#   x(k) & = 0.95 x(k-1) + \eta(k)\\
#   y(k) & = x(k) + \epsilon(k)
# \end{align}
# with $\eta(k) \sim \mathcal{N}\left(0,Q^{true}=1\right)$ and $\epsilon(k) \sim \mathcal{N}\left(0,R^{true}=1\right)$. Use the function *random.normal* to generate the Gaussian noises. Then, plot the $x$ and $y$ time series.

using Plots, Random, LinearAlgebra, Distributions, DataAssim

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
scatter!(y,linecolor=:black, markersize=2,label="Noisy observations")
ylims!(minimum(y)-1,maximum(y)+1)
title!("Simulated data from a linear Gaussian state-space model", fontsize=20)
# -

# Now, we apply the Kalman smoother using the true parameters: $M=0.95$, $H=1$, $Q=Q^{true}$ and $R=R^{true}$. We plot the corresponding results. The estimated state using Kalman is close to the truth and the $95\%$ confidence interval seems realistic.

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
xfree, yfree = FreeRun(gaussian, xi, Q, H, nmax, no)
# assimilation
xa, Pa = KalmanFilter(xi, Pi, gaussian, Q, yo, R, H, nmax, no);
# -

size(xa), size(Pa)

plot(vec(xt))
scatter!(vec(yo); markersize=2)
plot!(vec(xa), ribbon=[1.96*Pa[1,1,:],1.96*Pa[1,1,:]])
