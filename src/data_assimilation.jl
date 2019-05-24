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
                
        xb = xt.values[1]
        B  = 0.1 * Matrix(I, xt.nv, xt.nv)
        H  = Matrix( I, xt.nv, xt.nv)
        R  = sigma2 .* H

        new{T}(  method, np, xb, B, H, R, m )

    end

end
# -

struct Xhat

    part    :: Array{Array{Float64, 2}}
    weights :: Array{Array{Float64, 1}}
    values  :: Array{Array{Float64, 1}}
    loglik  :: Array{Float64, 1}
    time    :: Array{Float64, 1}

    function Xhat( x :: TimeSeries, np :: Int64)

        nt      = x.nt
        nv      = x.nv
        time    = x.time
        part    = [zeros(Float64,nv,np)   for i in 1:nt]
        weights = [ones(Float64,np) ./ np for i in 1:nt]
        values  = [zeros(Float64,nv)      for i in 1:nt]
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
    nv = yo.nv        # number of variables (dimensions of problem)

    # initialization
    x̂ = Xhat( yo, np )

    m_xa_part     = [zeros(Float64,(nv,np)) for i = 1:nt]
    xf_part       = [zeros(Float64,(nv,np)) for i = 1:nt] 
    pf            = [zeros(Float64,(nv,nv)) for i = 1:nt]
    xf            = zeros(Float64, (nv,np))
    m_xa_part_tmp = similar(xf)
    xf_mean       = similar(xf)
    ef            = similar(xf)

    @show da.xb

    for k in 1:nt
        # update step (compute forecasts)            
        if k == 1
            xf .= rand(MvNormal(da.xb, da.B), np)
        else
            xf, xf_mean    =  da.m(x̂.part[k-1])
            m_xa_part_tmp .=  xf_mean
            m_xa_part[k]  .=  xf_mean
        end

        xf_part[k] .= xf

        ef    .= xf * (Matrix(I, np, np) .- 1/np)
        pf[k] .= (ef * ef') ./ (np-1)

        # analysis step (correct forecasts with observations)          
        ivar_obs = findall(.!isnan.(yo.values[k]))
        n = length(ivar_obs)

        loglik = 0.0 :: Float64

        if n > 0
            μ   = zeros(Float64, n)
            σ   = da.R[ivar_obs,ivar_obs]
            eps = rand(MvNormal( μ, σ), np)
            yf  = da.H[ivar_obs,:] * xf
            SIGMA  = (da.H[ivar_obs,:] * pf[k]) * da.H[ivar_obs,:]' 
            SIGMA += da.R[ivar_obs,ivar_obs]
            SIGMA_INV = inv(SIGMA)
            K = (pf[k] * da.H[ivar_obs,:]') * SIGMA_INV 
            d = yo.values[k][ivar_obs] .- yf .+ eps
            x̂.part[k] .= xf .+ (d' * K')'
            # compute likelihood
            innov_ll = mean(yo.values[k][ivar_obs] .- yf, dims=2)
            loglik   = ( -0.5 * dot((innov_ll' * SIGMA_INV), innov_ll) 
                         -0.5 * (nv * log(2π) + log(det(SIGMA))))
        else
            x̂.part[k] .= xf
        end

        x̂.values[k] .= vec(sum(x̂.part[k] .* x̂.weights[k]',dims=2))
        x̂.loglik[k]  = loglik

    end 

x_hat1 = [[ -6.51351549 -11.37755422  13.60281104]
 [ -6.74929877 -11.89452898  13.4774771 ]
 [ -6.9165589  -12.17223569  13.63760501]
 [ -8.22471375 -13.88124435  16.11290499]
 [ -6.31150438 -11.26152994  12.61590882]]
x_hat2 = [[ -6.02865248 -10.5830217   13.27795867]
 [ -6.25160118 -11.05158806  13.07714473]
 [ -6.41491497 -11.31479942  13.20882905]
 [ -7.68178756 -13.03847079  15.4589259 ]
 [ -5.83342541 -10.42707995  12.29093051]]

    @show x̂.part[nt] = x_hat1'
    @show x̂.part[nt-1] = x_hat2'
    
    for k in nt:-1:1          
        if k == nt
            x̂.part[k] .= x̂.part[nt]
        else
            @show m_xa_part_tmp = m_xa_part[k+1]
m_xa_part_t= [[ -9.62769992 -14.39206914  21.1974462 ]
 [ -8.72415702 -13.75833961  18.77835229]
 [ -9.45835512 -14.40235001  20.46390493]
 [-10.10526007 -14.79825639  22.16267275]
 [ -9.20961258 -14.25456195  19.76245743]]
            m_xa_part_tmp = m_xa_part_t'
            #tej, m_xa_tmp = da.m(mean(x̂.part[k],dims=2))
            @show m_xa_tmp = [ -6.93970067; -12.12564896;  13.88381913]
            @show tmp1 = (x̂.part[k] .- mean(x̂.part[k],dims=2))'
            @show m_xa_part_tmp 
            tmp2 = m_xa_part_tmp .- m_xa_tmp
            @show tmp2
            pf[k+1] = [[ 0.52055832  0.86229172 -0.56650255]
 [ 0.86229172  1.42974479 -0.93394425]
 [-0.56650255 -0.93394425  0.63222399]]'
            @show pf[k+1]
            Ks   = 1.0 ./(np-1) .* ((tmp1' * tmp2') * inv_using_SVD(pf[k+1],0.9999))
            @show Ks
            xf_part[k+1] .= [[ -8.66705872 -13.61401191  18.93340205]
 [ -8.62838071 -13.62311946  18.74206282]
 [ -8.86092185 -13.78486135  19.29875258]
 [ -9.42985617 -14.32637588  20.56881391]
 [ -8.78387185 -13.73975828  19.15235668]]'
            @show x̂.part[k+1]
            @show xf_part[k+1]
            @show x̂.part[k] .+= ((x̂.part[k+1] .- xf_part[k+1])' * Ks')'
            return
        end
        @show  x̂.weights[k]
        x̂.values[k] .= vec(sum(x̂.part[k] .* x̂.weights[k]', dims=2))
        @show  x̂.values[k]
    end
    
    x̂       

end
# -


