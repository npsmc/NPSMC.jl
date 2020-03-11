# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
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

using LinearAlgebra, Distributions, DifferentialEquations

A = [2.50430061e-01 5.27015617e-02 8.98078667e-01 4.40716511e-01;
     6.26357655e-01 6.73178687e-01 2.24530912e-01 8.18640720e-01;
     6.85277570e-04 4.11289803e-01 4.45727050e-02 1.62456188e-01;
     8.52668155e-01 4.17796897e-01 8.18811326e-01 4.17691378e-01]

pinv(A)

"Returns the square root matrix by SVD"
function sqrt_svd(A)  
   F = svd(A)
   F.U * diagm(sqrt.(F.S)) * F.Vt
end


"Returns the Root Mean Squared Error"
RMSE(E) = sqrt.(mean(E.^2))




# +
using Distributions, Random

d = MvNormal(1.0 .* Matrix(I,3,3))

rng = MersenneTwister(42)

rand(rng, d)
# -

d = MvNormal(1.0 .* Matrix(I,3,3))



function gen_truth(f, x0, nt, Q, rng)
     nv = size(x0)
     xt = [zeros(nv) for i in 1:nt+1]
     xt[1] .= x0
     d = MvNormal(Q)
     for k in 1:nt
        xt[k+1] = f(xt[k]) .+ rand(rng, d)
     end
     xt
end


function gen_obs(h, xt, R, nb_assim, rng)
    nv = size(R)[1]
    nt = length(xt)
    yo = [zeros(nv) for i in 1:nt]
    yo .= NaN
    d = MvNormal(R)
    for k in 1:nt
        if k % nb_assim == 0
            yo[k] .= h(xt[k+1]) .+ rand(rng, R)
        end
    end
    return yo
end


# compute u0 to be in the attractor space
u0    = [8.0;0.0;30.0]
tspan = (0.0,5.0)
prob  = ODEProblem(ssm.model, u0, tspan, parameters)
u0    = last(solve(prob, reltol=1e-6, save_everystep=false))

# +
def _EnKS(Nx, Ne, T, H, R, Yo, Xt, No, xb, B, Q, alpha, f, prng):
  Xa, Xf = _EnKF(Nx, T, No, xb, B, Q, R, Ne, alpha, f, H, Yo, prng)

  Xs = np.zeros([Nx, Ne, T+1])
  Xs[:,:,-1] = Xa[:,:,-1]
  for t in range(T-1,-1,-1):
    Paf = np.cov(Xa[:,:,t], Xf[:,:,t])[:Nx, Nx:] ### MODIF PIERRE ###
    Pff = np.cov(Xf[:,:,t])
    try:
      K = Paf.dot(inv(Pff))
    except:
      K = Paf.dot(Pff**(-1)) ### MODIF PIERRE ###
    Xs[:,:,t] = Xa[:,:,t] + K.dot(Xs[:,:,t+1] - Xf[:,:,t])
   # for i in range(Ne):
   #   Xs[:,i, t] = Xa[:,i,t] + K.dot(Xs[:,i,t+1] - Xf[:,i,t])

  return Xs, Xa, Xf

def EnKS(params, prng):
  Nx = params['state_size']
  Ne = params['nb_particles']
  T  = params['temporal_window_size']
  H  = params['observation_matrix']
  R  = params['observation_noise_covariance']
  Yo = params['observations']
  Xt = params['true_state']
  No = params['observation_size']
  xb = params['background_state']
  B  = params['background_covariance']
  Q  = params['model_noise_covariance']
  alpha = params['inflation_factor']
  f  = params['model_dynamics']

  Xs, Xa, Xf = _EnKS(Nx, Ne, T, H, R, Yo, Xt, No, xb, B, Q, alpha, f, prng)

  res = {
          'smoothed_ensemble': Xs,
          'analysis_ensemble': Xa,
          'forecast_ensemble': Xf,
          'loglikelihood'    : _likelihood(Xf, Yo, H, R),
          'RMSE'             : RMSE(Xt - Xs.mean(1)),
          'params'           : params
         }
  return res
# -


function maximize(X, obs, H, f; structQ = :full, baseQ = nothing):
    
    Nx, Ne, T = X.shape
    No = obs.shape[0]

    xb = np.mean(X[:,:,0], 1)
    B = np.cov(X[:,:,0])

    sumSig = np.zeros((Nx, Ne, T-1))
    for t in range(T-1):
      sumSig[...,t] = X[...,t+1] - f(X[...,t])
    sumSig = np.reshape(sumSig, (Nx, (T-1)*Ne))
    sumSig = sumSig.dot(sumSig.T) / Ne
    if structQ == 'full':
      Q = sumSig/(T-1)
    elif structQ == 'diag':
      Q = np.diag(np.diag(sumSig))/T
    elif structQ == 'const':
      alpha = np.trace(pinv(baseQ).dot(sumSig)) / ((T-1)*Nx)
      Q = alpha*baseQ

    W = np.zeros([No, Ne, T-1])
    nobs = 0
    for t in range(T-1):
      if not np.isnan(obs[0,t]):
        nobs += 1
        W[:,:,t] = np.tile(obs[:,t], (Ne, 1)).T - H.dot(X[:,:,t+1])
    W = np.reshape(W, (No, (T-1)*Ne))
    R = W.dot(W.T) / (nobs*Ne)

    return xb, B, Q, R
end


def _likelihood(Xf, obs, H, R):
  T = Xf.shape[2]

  x = np.mean(Xf, 1)

  l = 0
  for t in range(T):
    if not np.isnan(obs[0,t]):
      innov = obs[:, t] - H.dot(x[:, t])
      Y = H.dot(Xf[:, :, t])

      sig = np.cov(Y) + R
      l -= .5 * np.log(np.linalg.det(sig)) 
      l -= .5 * innov.T.dot(inv(sig)).dot(innov)
  return l


def _EnKF(Nx, T, No, xb, B, Q, R, Ne, alpha, f, H, obs, prng):
  sqQ = sqrt_svd(Q)
  sqR = sqrt_svd(R)
  sqB = sqrt_svd(B)

  Xa = np.zeros([Nx, Ne, T+1])
  Xf = np.zeros([Nx, Ne, T])

  # Initialize ensemble
  for i in range(Ne):
    Xa[:,i,0] = xb + sqB.dot(prng.normal(size=Nx))

  for t in range(T):
    # Forecast
    # for i in range(Ne):
    #   Xf[:,i,t] = f(Xa[:,i,t]) + sqQ.dot(prng.normal(size=Nx))
    Xf[:,:,t] = f(Xa[:,:,t]) + sqQ.dot(prng.normal(size=(Nx, Ne)))
    Y = H.dot(Xf[:,:,t]) + sqR.dot(prng.normal(size=(No, Ne)))

    # Update
    if np.isnan(obs[0,t]):
      Xa[:,:,t+1] = Xf[:,:,t]
    else:
      Pfxx = np.cov(Xf[:,:,t])
      K = Pfxx.dot(H.T).dot(pinv(H.dot(Pfxx).dot(H.T) + R/alpha))
      innov = np.tile(obs[:,t], (Ne, 1)).T - Y
      Xa[:,:,t+1] = Xf[:,:,t] + K.dot(innov)
#      for i in range(Ne):
#        innov = obs[:,t] - Y[:,i]
#        Xa[:,i,t+1] = Xf[:,i,t] + K.dot(innov)

  return Xa, Xf


def EM_EnKS(params, prng):
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




# generate true state 
dt = .01 # integration time
sigma = 10;rho = 28;beta = 8./3 # physical parameters of Lorenz-63
m = lambda x: l63_predict(x, dt, sigma, rho, beta) # dynamic operator
h = lambda x: x # observation operator (nonlinear version)
H = eye(3) # observation operator (linear version)
K = 1000
Q_true = 0.05*eye(3)
x_true = gen_truth(m, r_[6.39435776, 9.23172442, 19.15323224], K, Q_true, RandomState(5))

# +
# generate true state 
dt = .01 # integration time
sigma = 10;rho = 28;beta = 8./3 # physical parameters of Lorenz-63
m = lambda x: l63_predict(x, dt, sigma, rho, beta) # dynamic operator
h = lambda x: x # observation operator (nonlinear version)
H = eye(3) # observation operator (linear version)
K = 1000
Q_true = 0.05*eye(3)
x_true = gen_truth(m, r_[6.39435776, 9.23172442, 19.15323224], K, Q_true, RandomState(5))

# generate noisy observations
dt_obs = 5 # 1 observation every dt_obs time steps
R_true = 2*eye(3)
y = gen_obs(h, x_true, R_true, dt_obs, RandomState(5))

# plot results
figure()
line1,=plot(range(K),x_true[0,1:],'r',linewidth=2)
line2,=plot(range(K),y[0,:],'.k')
legend([line1, line2], ['True state $x$', 'Noisy observations $y$'], prop={'size': 20})
title('Simulated data from the Lorenz-63 model (only the first component)', fontsize=20)

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




