using LinearAlgebra
using ProgressMeter

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
        k           = 1
        k_count     = 0
        m_xa_traj   = []
        weights_tmp = zeros(Float64, np)
        xf          = rand(MvNormal(da.xb, da.B), np)
        @show ivar_obs    = findall(.!isnan.(yo.u[k]))
      

        if length(ivar_obs) > 0
            # weights
            for ip in 1:np
                weights_tmp[ip] = pdf(yo.u[k][ivar_obs],
                                  da.H[ivar_obs,:] * xf[:,ip]',
                                  da.R[ivar_obs,ivar_obs])
            end
            # normalization
            @show weights_tmp ./= sum(weights_tmp)
            # resampling
            indic = resampleMultinomial(weights_tmp)
            x̂.part[k] = xf[indic,:]
            weights_tmp_indic = weights_tmp[indic]/sum(weights_tmp[indic])
            x̂.u[k] = sum(xf[indic,:]*weights_tmp_indic[np.newaxis],0)
            # find number of iterations before new observation
            k_count_end = minimum(findall(sum(.!isnan.(yo.u[k+1:end]),dims=2) .>= 1))

        else
            # weights
            weights_tmp .= 1.0 / np
            # resampling
            indic = resampleMultinomial(weights_tmp)
        end

        x̂.weights[:,k] = weights_tmp_indic
        
        for k in 2:nt
            # update step (compute forecasts) and add small Gaussian noise
            xf, tej = da.m(x̂.part[k-1]) + rand(zeros(nv),da.B ./ 100.0, np)        
            if k_count < length(m_xa_traj)
                m_xa_traj[k_count] = xf
            else
                push!(m_xa_traj, xf)
            end
            k_count += 1

            # analysis step (correct forecasts with observations)
            ivar_obs = findall(.!isnan.(yo.u[k,:]))
            if length(ivar_obs) > 0
                # weights
                for i in 1:np
                    weights_tmp[i] = pdf(yo.u[k][ivar_obs],
                                     da.H[:,ivar_obs] * xf[:,i]',
                                     da.R[ivar_obs,ivar_obs])
                end
                # normalization
                weights_tmp ./= sum(weights_tmp)
                # resampling
                indic = resampleMultinomial(weights_tmp)
                # stock results
                x̂.part[k-k_count_end:k+1] = m_xa_traj[:,indic,:]
                weights_tmp_indic = weights_tmp[indic]/sum(weights_tmp[indic])            
                x̂.u[k-k_count_end:k+1] = sum(m_xa_traj[:,indic,:] 
                                           .* weights_tmp_indic[np.newaxis],1)
                k_count = 0
                # find number of iterations  before new observation
                try
                    k_count_end = minimum(findall(sum(.!isnan.(yo.u[:,k+1:end]),dims=2) >= 1))
                catch ValueError
                    nothing
                end
            else
                # stock results
                x̂.part[k] = xf
                x̂.u[k]    = sum(xf .* weights_tmp_indic', dims=1)
            end
            # stock weights
            x̂.weights[k,:] = weights_tmp_indic 

        end

    else

        @error " $(da.method) not in [:EnKF, :EnKS, :PF] "

    end
    
    x̂       

end
