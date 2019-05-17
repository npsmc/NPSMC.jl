using LinearAlgebra

export DataAssimilation

"""
parameters of the filtering method
 - method :chosen method ('AnEnKF', 'AnEnKS', 'AnPF')
 - N : number of members (AnEnKF/AnEnKS) or particles (AnPF)
"""
struct DataAssimilation{T}

    method :: Symbol
    N      :: Int64
    xb     :: Vector{T}
    B      :: Array{Float64, 2}
    H      :: Array{Bool,    2}
    R      :: Array{Float64, 2}
    m      :: AnalogForecasting

    function DataAssimilation( m      :: AnalogForecasting,
                               method :: Symbol, 
                               N      :: Int64, 
                               xt     :: TimeSeries{T},
                               sigma2 :: Float64 ) where T
                
        xb = xt.values[1,:]
        B  = 0.1 * Matrix(I, xt.nv, xt.nv)
        H  = Matrix( I, xt.nv, xt.nv)
        R  = sigma2 .* H

        new{T}(  method, N, xb, B, H, R, m )

    end

    
end

struct Xhat

    part    :: Array{Float64, 3}
    weights :: Array{Float64, 2}
    values  :: Array{Float64, 2}
    loglik  :: Array{Float64, 1}
    time    :: Array{Float64, 1}

    function Xhat( time :: Vector{Float64}, N :: Int64, n :: Int64)

        T = length(time)

        part    = zeros(Float64, (T,N,n))
        weights = zeros(Float64, (T,N))
        values  = zeros(Float64, (T,n))
        loglik  = zeros(Float64, T)

        new( part, weights, values, loglik, time)

    end
end
    
""" 
    data_assimilation( yo, da)

Apply stochastic and sequential data assimilation technics using 
model forecasting or analog forecasting. 
"""
function data_assimilation(yo :: TimeSeries, da :: DataAssimilation)

    # dimensions
    n = length(da.xb)
    T, p = size(yo.values)
    # check dimensions
    @assert p == size(da.R)[0]

    # initialization
    x̂ = Xhat( yo.time, N, n )

    m_xa_part = zeros(Float64, (T,da.N,n))
    xf_part   = zeros(Float64, (T,da.N,n))
    Pf        = zeros(Float64, (T,n,n))

    for k in 1:T
        # update step (compute forecasts)            
        if k == 0
            xf = np.random.multivariate_normal(da.xb, da.B, da.N)
        else
            xf, m_xa_part_tmp = da.m(x̂.part[k-1,:,:])
            m_xa_part[k,:,:] = m_xa_part_tmp         
        end

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
            K = np.dot(np.dot(Pf[k,:,:],da.H[i_var_obs,:].T),SIGMA_INV)
            d = yo.values[k,i_var_obs]' .+ eps .- yf
            x̂.part[k,:,:] .= xf .+ (d * K.T)
            # compute likelihood
            innov_ll = mean(yo.values[k,i_var_obs]' .- yf,dims=1)
            loglik = -0.5*((innov_ll' * SIGMA_INV) * innov_ll) .- 0.5*(n*log.(2*pi)+log.(det(SIGMA)))
        else
            x̂.part[k,:,:] .= xf
        end

        x̂.weights[k,:] .= 1.0/da.N
        x̂.values[k,:]  .= sum(x̂.part[k,:,:]*x̂.weights[k,:,np.newaxis],dims=1)
        x̂.loglik[k]    = loglik

    end 
    
    
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
    
    x̂       

end
