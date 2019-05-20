# -*- coding: utf-8 -*-
using LinearAlgebra

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
        values  = zeros(Float64, (nt,np))
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
    pf            = zeros(Float64, (nt,np,np))
    m_xa_part_tmp = zeros(Float64, (np,nv))
    xf            = zeros(Float64, (np,nv))

    # Intiatize multivariate distribution
    d  = MvNormal(da.xb, da.B)

    for k in 1:1
        # update step (compute forecasts)            
        if k == 1
            xf = da.xb' .+ rand(d, np)'
        else
            xf                .=  da.m(x̂.part[k,:,:])
            m_xa_part_tmp     .=  xf
            m_xa_part[k,:,:]  .=  xf
        end

        xf_part[k,:,:] .= xf

        ef = xf' * (Matrix(I, np, np) .- 1 / np)

        pf[k,:,:] .= (ef' * ef) ./ (np-1)
        # analysis step (correct forecasts with observations)          
        i_var_obs = findall(.!isnan.(yo.values[k,:]))
        return i_var_obs

#        if length(i_var_obs)>0
#            d   = MvNormal( da.R[i_var_obs,i_var_obs],da.N)
#            eps = rand(d, da.N)
#            yf = (da.H[i_var_obs,:] * xf.T)'
#            SIGMA = ((da.H[i_var_obs,:] * Pf[k,:,:]) * da.H[i_var_obs,:].T)+da.R[i_var_obs,i_var_obs]
#            SIGMA_INV = inv(SIGMA)
#            K = np.dot(np.dot(Pf[k,:,:],da.H[i_var_obs,:].T),SIGMA_INV)
#            d = yo.values[k,i_var_obs]' .+ eps .- yf
#            x̂.part[k,:,:] .= xf .+ (d * K.T)
#            # compute likelihood
#            innov_ll = mean(yo.values[k,i_var_obs]' .- yf,dims=1)
#            loglik = -0.5*((innov_ll' * SIGMA_INV) * innov_ll) .- 0.5*(n*log.(2*pi)+log.(det(SIGMA)))
#        else
#            x̂.part[k,:,:] .= xf
#        end
#
#        x̂.weights[k,:] .= 1.0/da.N
#        x̂.values[k,:]  .= sum(x̂.part[k,:,:]*x̂.weights[k,:,np.newaxis],dims=1)
#        x̂.loglik[k]    = loglik
#
    end 
    
    
#    for k in T:-1:1          
#        if k == T
#            x̂.part[k,:,:] .= x̂.part[T,:,:]
#        else
#            m_xa_part_tmp = m_xa_part[k+1,:,:]
#            tej, m_xa_tmp = da.m(mean(x̂.part[k,:,:],dims=1))
#            tmp_1 =(x̂.part[k,:,:] .- mean(x̂.part[k,:,:],dims=1))
#            tmp_2 = m_xa_part_tmp .- m_xa_tmp
#            Ks = 1.0/(da.N-1) * ((tmp_1 * tmp_2) * inv_using_SVD(Pf[k+1,:,:],0.9999))
#            x̂.part[k,:,:] .+= (x̂.part[k+1,:,:] .- xf_part[k+1,:,:]) * Ks'
#        end
#        x̂.values[k,:] = sum(x̂.part[k,:,:] .* x̂.weights[k,:]',dims=1)
#    end
    
    x̂       

end
# -


