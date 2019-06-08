export EnKS

struct EnKS 

    np :: Int64

end

""" 
    data_assimilation( yo, da)

Apply stochastic and sequential data assimilation technics using 
model forecasting or analog forecasting. 
"""
function ( da :: DataAssimilation)( yo :: TimeSeries, mc :: EnKS )

    # dimensions
    nt = yo.nt        # number of observations
    np = mc.np        # number of particles
    nv = yo.nv        # number of variables (dimensions of problem)

    # initialization
    x̂ = TimeSeries( nt, nv )

    m_xa_part     = [zeros(Float64,(nv,np)) for i = 1:nt]
    xf_part       = [zeros(Float64,(nv,np)) for i = 1:nt] 
    part          = [zeros(Float64,(nv,np)) for i = 1:nt] 
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
            xf, xf_mean    =  da.m(part[k-1])
            m_xa_part_tmp .=  xf_mean
            m_xa_part[k]  .=  xf_mean
        end

        xf_part[k] .= xf

        ef    .= xf * (Matrix(I, np, np) .- 1/np)
        pf[k] .= (ef * ef') ./ (np-1)

        # analysis step (correct forecasts with observations)          
        ivar_obs = findall(.!isnan.(yo.u[k]))
        n = length(ivar_obs)

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
            part[k] .= xf .+ K * d

        else

            part[k] .= xf

        end

        x̂.u[k] .= vec(sum(part[k] ./ np ,dims=2))

    end 

    @showprogress -1 for k in nt:-1:1          

        if k == nt
            part[k] .= part[nt]
        else
            m_xa_part_tmp = m_xa_part[k+1]
            tej, m_xa_tmp = da.m(mean(part[k],dims=2))
            tmp1 = (part[k] .- mean(part[k],dims=2))'
            tmp2 = m_xa_part_tmp .- m_xa_tmp
            Ks  .= ((tmp1' * tmp2') * inv_using_SVD(pf[k+1],.9999))./(np-1)
            part[k] .+= Ks * (part[k+1] .- xf_part[k+1])
        end
        x̂.u[k] .= vec(sum(part[k] ./ np, dims=2))

    end

    x̂

end
