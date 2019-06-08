# -*- coding: utf-8 -*-
using LinearAlgebra
using ProgressMeter
using Distributions

export DataAssimilation

"""
    DataAssimilation( forecasting, method, np, xt, sigma2) 

parameters of the filtering method
 - method :chosen method (:AnEnKF, :AnEnKS, :AnPF)
 - N      : number of members (AnEnKF/AnEnKS) or particles (AnPF)
"""
mutable struct DataAssimilation

    method :: Symbol
    np     :: Int64
    xb     :: Vector{Float64}
    B      :: Array{Float64, 2}
    H      :: Array{Bool,    2}
    R      :: Array{Float64, 2}
    m      :: AbstractForecasting

    function DataAssimilation( m      :: AbstractForecasting,
                               method :: Symbol, 
                               np     :: Int64, 
                               xt     :: TimeSeries,
                               sigma2 :: Float64 )
                
        xb = xt.u[1]
        B  = 0.1 * Matrix(I, xt.nv, xt.nv)
        H  = Matrix( I, xt.nv, xt.nv)
        R  = sigma2 .* H

        new(  method, np, xb, B, H, R, m )

    end

end


mutable struct Xhat <: AbstractTimeSeries

    t       :: Array{Float64, 1}
    u       :: Array{Array{Float64, 1}}
    part    :: Array{Array{Float64, 2}}
    weights :: Array{Array{Float64, 1}}
    loglik  :: Array{Float64, 1}

    function Xhat( x :: TimeSeries, np :: Int64)

        nt      = x.nt
        nv      = x.nv
        time    = x.t
        part    = [zeros(Float64, nv, np)   for i in 1:nt]
        weights = [ones( Float64, np) ./ np for i in 1:nt]
        values  = [zeros(Float64, nv)       for i in 1:nt]
        loglik  =  zeros(Float64, nt)

        new( time, values, part, weights, loglik)

    end
end


export data_assimilation

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

    # EnKF and EnKS methods
    if (da.method == :EnKF || da.method == :EnKS)

        m_xa_part     = [zeros(Float64,(nv,np)) for i = 1:nt]
        xf_part       = [zeros(Float64,(nv,np)) for i = 1:nt] 
        pf            = [zeros(Float64,(nv,nv)) for i = 1:nt]
        xf            = zeros(Float64, (nv,np))
        m_xa_part_tmp = similar(xf)
        xf_mean       = similar(xf)
        ef            = similar(xf)
        Ks            = zeros(Float64,(3,3))

        @showprogress 1 for k in 1:nt

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
            ivar_obs = findall(.!isnan.(yo.u[k]))
            n = length(ivar_obs)

            loglik = 0.0 :: Float64

            if n > 0
                μ   = zeros(Float64, n)
                σ   = da.R[ivar_obs,ivar_obs]
                eps = rand(MvNormal( μ, σ), np)
                yf  = da.H[ivar_obs,:] * xf
                SIGMA     = (da.H[ivar_obs,:] * pf[k]) * da.H[ivar_obs,:]' 
                SIGMA   .+= da.R[ivar_obs,ivar_obs]
                SIGMA_INV = inv(SIGMA)
                K  = (pf[k] * da.H[ivar_obs,:]') * SIGMA_INV 
                d  = yo.u[k][ivar_obs] .- yf .+ eps
                x̂.part[k] .= xf .+ K * d
                # compute likelihood
                innov_ll = mean(yo.u[k][ivar_obs] .- yf, dims=2)
                loglik   = ( -0.5 * dot((innov_ll' * SIGMA_INV), innov_ll) 
                             -0.5 * (nv * log(2π) + log(det(SIGMA))))
            else
                x̂.part[k] .= xf
            end

            x̂.u[k] .= vec(sum(x̂.part[k] .* x̂.weights[k]',dims=2))
            x̂.loglik[k]  = loglik

        end 

        if da.method == :EnKS

            @showprogress -1 for k in nt:-1:1          

                if k == nt
                    x̂.part[k] .= x̂.part[nt]
                else
                    m_xa_part_tmp = m_xa_part[k+1]
                    tej, m_xa_tmp = da.m(mean(x̂.part[k],dims=2))
                    tmp1 = (x̂.part[k] .- mean(x̂.part[k],dims=2))'
                    tmp2 = m_xa_part_tmp .- m_xa_tmp
                    Ks  .= ((tmp1' * tmp2') * inv_using_SVD(pf[k+1],.9999))./(np-1)
                    x̂.part[k] .+= Ks * (x̂.part[k+1] .- xf_part[k+1])
                end
                x̂.u[k] .= vec(sum(x̂.part[k] .* x̂.weights[k]', dims=2))

            end

        end

    elseif da.method == :PF

        # special case for k=1
        k          = 1
        m_xa_traj  = Array{Float64,2}[]
        xf         = rand(MvNormal(da.xb, da.B), np)
        ivar_obs   = findall(.!isnan.(yo.u[k]))
        nobs       = length(ivar_obs)
        weights    = zeros(Float64, np)
        indic      = zeros(Int64, np)
      
        if nobs > 0

            for ip in 1:np
                μ = vec(da.H[ivar_obs,:] * xf[:,ip])
                σ = Matrix(da.R[ivar_obs,ivar_obs])
                d = MvNormal(μ, σ)
                weights[ip] = pdf(d, yo.u[k][ivar_obs])
            end
            # normalization
            weights ./= sum(weights)
            # resampling
            resample!(indic, weights)
            x̂.part[k]  .= xf[:,indic]
            weights    .= weights[indic] ./ sum(weights[indic])            
            x̂.u[k]     .= vec(sum(x̂.part[k] .* weights', dims=2))

            # find number of iterations before new observation
            # todo: try the findnext function
            # findnext(.!isnan.(vcat(yo.u'...)), k+1)
            knext = 1
            while knext+k <= nt && all(isnan.(yo.u[k+knext])) 
                knext +=1
            end

        else

            weights .= 1.0 / np # weights
            resample!(indic, weights) # resampling

        end

        x̂.weights[k] .= weights

        kcount = 1

        for k in 2:nt
            # update step (compute forecasts) and add small Gaussian noise
            xf, tej = da.m(x̂.part[k-1]) 
            xf .+= rand(MvNormal(zeros(nv), da.B ./ 100.0), np)
            if kcount <= length(m_xa_traj)
                m_xa_traj[kcount] .= xf
            else
                push!(m_xa_traj, xf)
            end
            kcount += 1

            # analysis step (correct forecasts with observations)
            ivar_obs = findall(.!isnan.(yo.u[k]))

            if length(ivar_obs) > 0
                # weights
                σ = Symmetric(da.R[ivar_obs,ivar_obs])
                for ip in 1:np
                    μ = vec(da.H[ivar_obs,:] * xf[:,ip])
                    d = MvNormal(μ, σ)
                    weights[ip] = pdf(d, yo.u[k][ivar_obs])
                end 
                # normalization
                weights ./= sum(weights)
                # resampling
                resample!(indic, weights)
                weights .= weights[indic] ./ sum(weights[indic])            
                # stock results
                for j in 1:knext
                    jm = k-knext+j
                    for ip in 1:np
                       x̂.part[jm][:,ip] .= m_xa_traj[j][:,indic[ip]]
                    end
                    x̂.u[jm] .= vec(sum( x̂.part[jm] .* weights', dims=2))
                end
                kcount = 1
                # find number of iterations  before new observation
                knext = 1
                while knext+k <= nt && all(isnan.(yo.u[k+knext])) 
                    knext +=1
                end
            else
                # stock results
                x̂.part[k] .= xf
                x̂.u[k]    .= vec(sum(xf .* weights', dims=2))
            end
            # stock weights
            x̂.weights[k] .= weights

        end


    else

        @error " $(da.method) not in [:EnKF, :EnKS, :PF] "

    end
    
    x̂

end
