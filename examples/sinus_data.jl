# # Sinus data

using LinearAlgebra
using NPSMC
using Plots

# ### Generate simulated data (Sinus Model)

# parameters
dx = 1                # dimension of the state
dt_int = 1.0               # fixed integration time
dt_model = 1                # chosen number of model time step
var_obs = [0]              # indices of the observed variables
dy = length(var_obs)  # dimension of the observations

function h(x) # observation model
    dx = length(x)
    H = Matrix(I, dx, dx)
    H .* x
end

jac_h(x) = x
const a = 3.0::Float64
mx(x) = sin.(a .* x)
jac_mx(x) = a .* cos.(a .* x)

# Setting covariances
sig2_Q = 0.1
sig2_R = 0.1

# prior state
x0 = [1.0]

sinus3 = NPSMC.SSM(h, jac_h, mx, jac_mx, dt_int, dt_model, x0, var_obs, sig2_Q, sig2_R)

# generate data
T = 2000# length of the training
X, Y = generate_data(sinus3, x0, T)

values = X[1]
scatter(values, mx(values))
scatter!(values[1:end-1], values[2:end], markersize = 2)
