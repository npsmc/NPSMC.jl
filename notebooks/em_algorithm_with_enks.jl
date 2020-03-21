# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.0
#   kernelspec:
#     display_name: Julia 1.3.1
#     language: julia
#     name: julia-1.3
# ---

using LinearAlgebra, Distributions, DifferentialEquations
using Plots, Random

include("../src/utils.jl")
include("../src/ensemble.jl")



# # Generate true state

# +
"""
    Time integration of Lorenz-63
    Runke-Kutta fourth order
"""
function integrate( model, x, p, dt )
        
    f1 = zero(x)
    f2 = zero(x)
    f3 = zero(x)
    f4 = zero(x)
    t = nothing
    ncy = 1000
    dtcy = dt / ncy
    
    for k = 1:ncy
        xtmp = x
        model(f1, xtmp, p, t)
        xtmp = x .+ f1 .* dtcy
        model(f2, xtmp, p, t)
        xtmp = x .+ f2 .* dtcy
        model(f3, xtmp, p, t)
        xtmp = x .+ f3 .* dtcy
        model(f4, xtmp, p, t)
        x .+= ( f1 .+ 2 .* f2 .+ 2 .* f3 .+ f4 ) ./ 6 * dtcy
    end
    
    x
     
end

# +
dt = .01 
sigma = 10
rho = 28
beta = 8/3 
x0 = [6.39435776, 9.23172442, 19.15323224]
nt, nv = 1000, 3
T = nt * dt

function lorenz63(du, u, p, t)

    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[2] - u[3]) - u[2]
    du[3] = u[1] * u[2] - p[3] * u[3]

end

parameters = [sigma, rho, beta]

function gen_truth(model, x0, nt, Q, rng)
     d = MvNormal(Q)
     xt = [zero(x0) for i in 1:nt+1]
     xt[1] .= x0
     @show dt
     for k in 1:nt        
        xt[k+1] .= integrate(model, xt[k], parameters, dt) .+ rand(rng, d)
     end
     xt
end
# -

Q_true = 0.05 .* Matrix(I, 3, 3)
x_true = gen_truth(lorenz63, x0, nt, Q_true, MersenneTwister(5));

# +
function gen_obs(h, xt, R, nb_assim, rng)
    nv = size(xt[1])[1]
    nt = length(xt)
    yo = [zeros(Float64,nv) for i in 1:nt]
    d = MvNormal(R)
    for k in 1:nt
        if k % nb_assim == 0
            yo[k] .= h(xt[k+1]) .+ rand(rng, R)
        else
            yo[k] .= NaN
        end
    end
    return yo
end

# generate noisy observations
h(x) = x # observation operator (nonlinear version)
H = 1.0 .* Matrix(I,3,3) # observation operator (linear version)
dt_obs = 5 # 1 observation every dt_obs time steps
R_true = 2.0 .* H
y = gen_obs(h, x_true, R_true, dt_obs, MersenneTwister(5));
# -

# # Plot results

# +
time = collect(0:dt:T)
obs = vcat(y'...)[:,1]
indices = findall(!isnan, obs )

plot(time, getindex.(x_true,1); lc=:red,lw=2, label="True state")
title!("Simulated data from the Lorenz-63 model (only the first component)")
scatter!(time[indices], obs[indices], mc = :blue, ms = 2,  label="Noisy observations")

# +
"""
    _EnKF(nv, nt, no, xb, B, Q, R, ne, alpha, f, H, obs, prng)
    
EnKF forecast and analysis steps
"""
function _EnKF(nv, nt, no, xb, B, Q, R, ne, alpha, f, H, obs, prng)

    sqQ = sqrt_svd(Q)
    sqR = sqrt_svd(R)
    sqB = sqrt_svd(B)

    xa = [zeros(nv,ne) for i in 1:nt+1] 
    xf = [zeros(nv,ne) for i in 1:nt+1]

    # Initialize ensemble
    for i in 1:ne
        xa[1][:,i] .= xb .+ sqB * randn(prng, nv)
    end

    for k in 1:nt
        Xf[k] .= f(Xa[k]) .+ sqQ * randn(prng, (nv,ne))
        Y = H * Xf[k] .+ sqR * randn(prng,(no, ne))

        if isnan(obs[k][1])
            xa[k+1] .= Xf[k]
        else
            Pfxx = cov(Xf[k])
            K = Pfxx * (H' * pinv(H * Pfxx * H' + R ./ alpha))
            innov = obs[k]' .- Y
            xa[k+1] = xf[k] + K * innov
        end
    end
              
    return xa, xf

end
# -

"""
    _EnKS(nv, ne, nt, H, R, yo, xt, no, xb, B, Q, alpha, f, prng)

"""
function _EnKS(nv, ne, nt, H, R, yo, xt, no, xb, B, Q, alpha, f, prng)

    xa, xf = _EnKF(nv, nt, no, xb, B, Q, R, ne, alpha, f, H, Yo, prng)

    xs = [zeros(nv,ne) for i in 1:nt+1] 
    xs[end] .= xa[end]
    for t in nt:-1:1
        Paf = cov(xa[t], xf[t])
        Pff = cov(xf[t])
        K = Paf * pinv(Pff)
        xs[t] .= xa[t] .+ K * (xs[t+1] - xf[t])
    end
    
    return xs, xa, xf
end


# +
"""
    EnKS(params, prng)
    
"""
function EnKS(params, prng)

  Nx = params.state_size
  Ne = params.nb_particles
  T  = params.temporal_window_size
  H  = params.observation_matrix
  R  = params.observation_noise_covariance
  Yo = params.observations
  Xt = params.true_state
  No = params.observation_size
  xb = params.background_state
  B  = params.background_covariance
  Q  = params.model_noise_covariance
  alpha = params.inflation_factor
  f  = params.model_dynamics

  xs, xa, xf = _EnKS(Nx, Ne, T, H, R, Yo, Xt, No, xb, B, Q, alpha, f, prng)

  res = (
          smoothed_ensemble = xs,
          analysis_ensemble = xa,
          forecast_ensemble = xf,
          loglikelihood =  _likelihood(xf, yo, H, R),
          RMSE = RMSE(Xt - mean(xs),
          params = params
         )
  res
  
end
# -

function maximize(X, obs, H, f; structQ = :full, baseQ = nothing)
    
    Nx, Ne, T = X.shape
    No = obs.shape[0]

    xb = mean(X[1], dims=2)
    B = cov(X[1])

    sumSig = [zeros(nv, ne) for i in 1:T-1]
    for t in 1:T-1
      sumSig[t] = X[t+1] - f(X[t])
    end
    sumSig = reshape(sumSig, Nx, (T-1)*Ne)
    sumSig = sumSig * sumSig' / Ne
    if structQ == :full
      Q = sumSig/(T-1)
    elseif structQ == :diag
      Q = diagm(diagm(sumSig)) ./ T
    elseif structQ == :const
      alpha = trace(pinv(baseQ) * sumSig) ./ ((T-1)*Nx)
      Q = alpha*baseQ
    end

    W = [zeros((No, Ne)) for i in 1:T-1]
    nobs = 0
    for t in range(T-1)
        if !isnan(obs[t][1])
            nobs += 1
            W[t] .= obs[t] .- H * X[t+1]
        end
    end
    W = vcat(W... ) # reshape(W, (No, (T-1)*Ne))
    R = W * W' / (nobs*Ne)

    return xb, B, Q, R
end

A = rand(3,10)
mean(A, dims=2)
# +
function _likelihood(xf, obs, H, R)

  nt = length(xf)

  x = mean(xf, 1)

  l = 0
  for t in range(T):
    if not np.isnan(obs[0,t]):
      innov = obs[:, t] - H.dot(x[:, t])
      Y = H.dot(Xf[:, :, t])

      sig = np.cov(Y) + R
      l -= .5 * np.log(np.linalg.det(sig)) 
      l -= .5 * innov.T.dot(inv(sig)).dot(innov)
  return l

end

# +
function EM_EnKS(params, prng):

  xb      = params['initial_background_state']
  B       = params['initial_background_covariance']
  Q       = params['initial_model_noise_covariance']
  R       = params['initial_observation_noise_covariance']
  f       = params['model_dynamics']
  H       = params['observation_matrix']
  Yo      = params['observations']
  Ne      = params['nb_particles']
  nIter   = params['nb_EM_iterations']
  Xt      = params['true_state']
  alpha   = params['inflation_factor']
  Nx      = params['state_size']
  T       = params['temporal_window_size']
  No      = params['observation_size']
  estimateQ  = params['is_model_noise_covariance_estimated']
  estimateR  = params['is_observation_noise_covariance_estimated']
  estimateX0 = params['is_background_estimated']
  structQ = params['model_noise_covariance_structure']
  if structQ == 'const':
    baseQ = params['model_noise_covariance_matrix_template']
  else:
    baseQ = None

  loglik = np.zeros(nIter)
  rmse_em = np.zeros(nIter)

  Q_all  = np.zeros(np.r_[Q.shape,  nIter+1])
  R_all  = np.zeros(np.r_[R.shape,  nIter+1])
  B_all  = np.zeros(np.r_[B.shape,  nIter+1])
  xb_all = np.zeros(np.r_[xb.shape, nIter+1])

  Q_all[:,:,0] = Q
  R_all[:,:,0] = R
  xb_all[:,0]  = xb
  B_all[:,:,0] = B

  for k in tqdm(range(nIter)):

    # E-step
    Xs, Xa, Xf = _EnKS(Nx, Ne, T, H, R, Yo, Xt, No, xb, B, Q, alpha, f, prng)
    loglik[k] = _likelihood(Xf, Yo, H, R)
    rmse_em[k] = RMSE(Xt - Xs.mean(1))

    # M-step
    xb_new, B_new, Q_new, R_new = _maximize(Xs, Yo, H, f, structQ=structQ, baseQ=baseQ)
    if estimateQ:
      Q = Q_new
    if estimateR:
      R = R_new
    if estimateX0:
      xb = xb_new
      B = B_new

    Q_all[:,:,k+1] = Q
    R_all[:,:,k+1] = R
    xb_all[:,k+1] = xb
    B_all[:,:,k+1] = B

  res = {
          'smoothed_ensemble'              : Xs,
          'EM_background_state'            : xb_all,
          'EM_background_covariance'       : B_all,
          'EM_model_noise_covariance'      : Q_all,
          'EM_observation_noise_covariance': R_all,
          'loglikelihood'                  : loglik,
          'RMSE'                           : rmse_em,
          'params'                         : params
        }
  return res

end
# -

# apply EnKS with good covariances
params = { 'state_size'                  : 3,
           'nb_particles'                : 100,
           'temporal_window_size'        : K,
           'observation_matrix'          : H,
           'observation_noise_covariance': R_true,
           'observations'                : y,
           'true_state'                  : x_true,
           'observation_size'            : 3,
           'background_state'            : r_[6.39435776, 9.23172442, 19.15323224],
           'background_covariance'       : eye(3),
           'model_noise_covariance'      : Q_true,
           'inflation_factor'            : 1,
           'model_dynamics'              : m}

EnKS_true_Q_R=EnKS(params, RandomState(5))
ens_true_Q_R=EnKS_true_Q_R['smoothed_ensemble'] # smoother ensembles
xs_true_Q_R=mean(EnKS_true_Q_R['smoothed_ensemble'], 1)
Ps_true_Q_R=diag(cov(ens_true_Q_R[0,:,:], rowvar=False))

# plot results
figure()
line1,=plot(range(K),x_true[0,1:],'r',linewidth=2)
line2,=plot(range(K),y[0,:],'.k')
line3,=plot(range(K),xs_true_Q_R[0,1:],'k',linewidth=2)
fill_between(range(K), squeeze(xs_true_Q_R[0,1:]) - 1.96 * sqrt(squeeze(Ps_true_Q_R[1:])),
             squeeze(xs_true_Q_R[0,1:]) + 1.96 * sqrt(squeeze(Ps_true_Q_R[1:])), color='k', alpha=.2)
legend([line1, line2, line3], ['True state $x$', 'Noisy observations $y$', 'Estimated state'], prop={'size': 20})
title('Results of the EnKS (only the first component)', fontsize=20)

# -

