# -*- coding: utf-8 -*-
using LinearAlgebra
using ProgressMeter

export DataAssimilation

# +
"""
parameters of the filtering method
 - method :chosen method ('AnEnKF', 'AnEnKS', 'AnPF')
 - N : number of members (AnEnKF/AnEnKS) or particles (AnPF)
"""
mutable struct DataAssimilation{T}

    method :: Symbol
    np     :: Int64
    xb     :: Vector{T}
    B      :: Array{Float64, 2}
    H      :: Array{Bool,    2}
    R      :: Array{Float64, 2}
    m      :: AbstractForecasting

    function DataAssimilation( m      :: AbstractForecasting,
                               method :: Symbol, 
                               np     :: Int64, 
                               xt     :: TimeSeries{T},
                               sigma2 :: Float64 ) where T
                
        xb = vec(xt.values[1,:])
        B  = 0.1 * Matrix(I, xt.nv, xt.nv)
        H  = Matrix( I, xt.nv, xt.nv)
        R  = sigma2 .* H

        new{T}(  method, np, xb, B, H, R, m )

    end

end
# -

struct Xhat

    part    :: Array{Float64, 3}
    weights :: Array{Float64, 2}
    values  :: Array{Float64, 2}
    loglik  :: Array{Float64, 1}
    time    :: Array{Float64, 1}

    function Xhat( x :: TimeSeries, np :: Int64)

        nt      = x.nt
        nv      = x.nv
        time    = x.time
        part    = zeros(Float64, (nt,np,nv))
        weights = zeros(Float64, (nt,np))
        values  = zeros(Float64, (nt,nv))
        loglik  = zeros(Float64,  nt)

        new( part, weights, values, loglik, time)

    end
end

# +
""" 
    data_assimilation( yo, da)

Apply stochastic and sequential data assimilation technics using 
model forecasting or analog forecasting. 
"""
function data_assimilation(yo :: TimeSeries, da :: DataAssimilation)

    # dimensions
    nt = yo.nt        # number of observations
    np = da.np        # number of particles
    nv = yo.nv        # number of variables

    # initialization
    x̂ = Xhat( yo, np )

    m_xa_part     = zeros(Float64, (nt,np,nv))
    xf_part       = zeros(Float64, (nt,np,nv))
    pf            = zeros(Float64, (nt,nv,nv))
    m_xa_part_tmp = zeros(Float64, (np,nv))
    xf            = zeros(Float64, (np,nv))


    @showprogress 1 for k in 1:nt
        # update step (compute forecasts)            
        if k == 1
            xf = da.xb' .+ rand(MvNormal(da.xb, da.B), np)'
        else
            xf          .=  da.m(x̂.part[k-1,:,:])
            m_xa_part_tmp     .=  xf
            m_xa_part[k,:,:]  .=  xf
        end

        xf_part[k,:,:] .= xf

        ef = xf' * (Matrix(I, np, np) .- 1/np)

        pf[k,:,:] .= (ef * ef') ./ (np-1)

        # analysis step (correct forecasts with observations)          
        i_var_obs = findall(.!isnan.(yo.values[k,:]))
        n = length(i_var_obs)

        if n > 0
            μ   = zeros(Float64, n)
            σ   = da.R[i_var_obs,i_var_obs]
            eps = rand(MvNormal( μ, σ), np)'
            yf  = (da.H[i_var_obs,:] * xf')'
            SIGMA  = (da.H[i_var_obs,:] * pf[k,:,:]) * da.H[i_var_obs,:]' 
            SIGMA += da.R[i_var_obs,i_var_obs]
            SIGMA_INV = inv(SIGMA)
            K = pf[k,:,:] * da.H[i_var_obs,:]' * SIGMA_INV 
            d = yo.values[k,i_var_obs] .+ eps .- yf
            x̂.part[k,:,:] = xf + d * K'
            # compute likelihood
            innov_ll = mean(yo.values[k,i_var_obs] .- yf, dims=1)
            loglik = -0.5 .* (dot((innov_ll' * SIGMA_INV), innov_ll) 
                        .- (n * log.(2*pi) .+ log(det(SIGMA))))
        else
            x̂.part[k,:,:] .= xf
            loglik         = 0.0
        end

        x̂.weights[k,:] .= 1.0/np
        x̂.values[k,:]  .= vec(sum(x̂.part[k,:,:] .* x̂.weights[k,:],dims=1))
        x̂.loglik[k]     = loglik

    end 
    
    @showprogress 1 for k in nt:-1:1          
        if k == nt
            x̂.part[k,:,:] .= x̂.part[nt,:,:]
        else
            m_xa_part_tmp = m_xa_part[k+1,:,:]
            tej, m_xa_tmp = da.m(mean(x̂.part[k,:,:],dims=1))
            tmp1 =(x̂.part[k,:,:] .- mean(x̂.part[k,:,:],dims=1))'
            tmp2 = m_xa_part_tmp .- m_xa_tmp
            Ks   = 1.0 ./(np-1) .* ((tmp1 * tmp2) * inv_using_SVD(pf[k+1,:,:],0.9999))
            x̂.part[k,:,:] .+= (x̂.part[k+1,:,:] .- xf_part[k+1,:,:]) * Ks'
        end
        x̂.values[k,:] .= vec(sum(x̂.part[k,:,:] .* x̂.weights[k,:], dims=1))
    end
    
    x̂       

end
# -


