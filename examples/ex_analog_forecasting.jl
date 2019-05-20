# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.2
#   kernelspec:
#     display_name: Julia 1.1.0
#     language: julia
#     name: julia-1.1
# ---

using LinearAlgebra
include("../src/models.jl")
include("../src/time_series.jl")
include("../src/state_space.jl")
include("../src/catalog.jl")
include("../src/generate_data.jl")

?StateSpaceModel

# +
σ = 10.0
ρ = 28.0
β = 8.0/3

dt_integration = 0.01
dt_states      = 1 
dt_obs         = 8 
parameters     = [σ, ρ, β]
var_obs        = [1]
nb_loop_train  = 30 
nb_loop_test   = 1
sigma2_catalog = 0.0
sigma2_obs     = 2.0

ssm = StateSpaceModel( dt_integration, dt_states, dt_obs, 
                       parameters, var_obs,
                       nb_loop_train, nb_loop_test,
                       sigma2_catalog, sigma2_obs )

xt, yo, catalog = generate_data( ssm )
# -
include("../src/model_forecasting.jl")
af = AnalogForecasting( 5, xt, catalog )
N = 10
da = DataAssimilation( af, :EnKs, N, xt, ssm.sigma2_obs)
# # +

# dimensions
@show n = length(da.xb)
@show T, p = size(yo.values)
# check dimensions
@assert p == size(da.R)[1]

# initialization
x̂ = Xhat( yo.time, N, n )

m_xa_part = zeros(Float64, (T,da.N,n))
xf_part   = zeros(Float64, (T,da.N,n))
Pf        = zeros(Float64, (T,n,n));

# +
for k in 1:1
    # update step (compute forecasts)            
    if k == 1
        d = MvNormal(da.xb, da.B)
        xf = rand(d, (da.N,n))
    else
        xf, m_xa_part_tmp = da.m(x̂.part[k,:,:])
        m_xa_part[k,:,:] = m_xa_part_tmp         
    end
    @show xf
    xf_part[k,:,:] .= xf
    Ef = xf' * (Matrix(I, da.N, da.N) .- 1/da.N)
    Pf[k,:,:] .= (Ef * Ef.T) ./ (da.N-1)
    # analysis step (correct forecasts with observations)          
    i_var_obs = findall(.!isnan.(values[k,:]))

    if length(i_var_obs)>0
        d   = MvNormal( da.R[i_var_obs,i_var_obs],da.N)
        eps = rand(d, da.N)
        yf = (da.H[i_var_obs,:] * xf.T)'
        SIGMA = ((da.H[i_var_obs,:] * Pf[k,:,:]) * da.H[i_var_obs,:].T)+da.R[i_var_obs,i_var_obs]
        SIGMA_INV = inv(SIGMA)
        K = (Pf[k,:,:] * da.H[i_var_obs,:]') * SIGMA_INV
        d = yo.values[k,i_var_obs]' .+ eps .- yf
        x̂.part[k,:,:] .= xf .+ (d * K')
        # compute likelihood
        innov_ll = mean(yo.values[k,i_var_obs]' .- yf,dims=1)
        loglik = -0.5*((innov_ll' * SIGMA_INV) * innov_ll) .- 0.5*(n*log.(2*pi)+log.(det(SIGMA)))
    else
        x̂.part[k,:,:] .= xf
    end

    x̂.weights[k,:] .= 1.0/da.N
    x̂.values[k,:]  .= sum(x̂.part[k,:,:]*x̂.weights[k,:,np.newaxis],dims=1)
    x̂.loglik[k]     = loglik

end 
# -


for k in T:-1:1
    if k == T
        x̂.part[k,:,:] .= x̂.part[T,:,:]
    else
        m_xa_part_tmp = m_xa_part[k+1,:,:]
        tej, m_xa_tmp = da.m(mean(x̂.part[k,:,:],dims=1))
        tmp_1 =(x̂.part[k,:,:] .- mean(x̂.part[k,:,:],dims=1))
        tmp_2 = m_xa_part_tmp .- m_xa_tmp
        Ks = 1.0/(da.N-1) * ((tmp_1 * tmp_2) * inv_using_SVD(Pf[k+1,:,:],0.9999))
        x̂.part[k,:,:] .+= (x̂.part[k+1,:,:] .- xf_part[k+1,:,:]) * Ks'
    end
    x̂.values[k,:] = sum(x̂.part[k,:,:] .* x̂.weights[k,:]',dims=1)
end

# -
