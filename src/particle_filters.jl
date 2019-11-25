export PF

struct PF <: MonteCarloMethod

    np :: Int64

end

""" 
    data_assimilation( yo, da, PF(100) )

Apply particle filters data assimilation technics using 
model forecasting or analog forecasting. 

"""
function ( da :: DataAssimilation )( yo :: TimeSeries, mc :: PF; progress = true )

    # dimensions
    nt = yo.nt        # number of observations
    np = mc.np        # number of particles
    nv = yo.nv        # number of variables (dimensions of problem)

    # initialization
    x̂ = TimeSeries( nt, nv )

    # special case for k=1
    k          = 1
    m_xa_traj  = Array{Float64,2}[]
    xf         = rand(MvNormal(da.xb, da.B), np)
    ivar_obs   = findall(.!isnan.(yo.u[k]))
    nobs       = length(ivar_obs)
    weights    = zeros(Float64, np)
    indic      = zeros(Int64, np)
    part       = [zeros(Float64,(nv,np)) for i = 1:nt] 
    
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
        part[k]  .= xf[:,indic]
        weights  .= weights[indic] ./ sum(weights[indic])            
        x̂.u[k]   .= vec(sum(part[k] .* weights', dims=2))

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

    kcount = 1

    if progress p = Progress(nt) end

    for k in 2:nt

        if progress next!(p) end

        # update step (compute forecasts) and add small Gaussian noise
        xf, tej = da.m(part[k-1]) 
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
                   part[jm][:,ip] .= m_xa_traj[j][:,indic[ip]]
                end
                x̂.u[jm] .= vec(sum( part[jm] .* weights', dims=2))
            end
            kcount = 1
            # find number of iterations  before new observation
            knext = 1
            while knext+k <= nt && all(isnan.(yo.u[k+knext])) 
                knext +=1
            end
        else
            # stock results
            part[k] .= xf
            x̂.u[k]  .= vec(sum(xf .* weights', dims=2))
        end

    end

    x̂

end
