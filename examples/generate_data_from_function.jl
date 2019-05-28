# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.3
#   kernelspec:
#     display_name: Julia 1.1.0
#     language: julia
#     name: julia-1.1
# ---

include("../src/models.jl")
include("../src/time_series.jl")
include("../src/state_space.jl")
include("../src/catalog.jl")
include("../src/plot.jl")
include("../src/generate_data.jl")

# ### GENERATE SIMULATED DATA (SINUS MODEL)

# +
# parameters
dx       = 1                # dimension of the state
dt_int   = 1.               # fixed integration time
dt_model = 1                # chosen number of model time step
var_obs  = [0]              # indices of the observed variables
dy       = length(var_obs)  # dimension of the observations

function h(x) # observation model
    dx = length(x)
    H = Matrix(I,dx,dx) 
    H .* x 
end
jac_h(x)  = x
const a = 3. :: Float64
mx(x)     = sin.(a .* x)
jac_mx(x) = a .* cos.( a .* x)

# Setting covariances
sig2_Q = 0.1
sig2_R = 0.1

# prior state
x0 = [1.]

sinus3   = SSM(h, jac_h, mx, jac_mx, dt_int, dt_model, x0, var_obs, sig2_Q, sig2_R)
# -

# generate data
T        = 2000# length of the training
X, Y     = generate_data( sinus3, x0, T)

using Plots

values = vcat(X.u'...)
scatter( values, mx(values))
scatter!(values[1:end-1], values[2:end], markersize=2)




