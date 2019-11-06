using NearestNeighbors, PDMats

export AnalogForecasting

"""
    AnalogForecasting(k, xt, catalog)

parameters of the analog forecasting method

- k            : number of analogs
- neighborhood : global analogs
- catalog      : catalog with analogs and successors
- regression   : (:locally_constant, :increment, :local_linear)
- sampling     : (:gaussian, :multinomial)

"""
struct AnalogForecasting <: AbstractForecasting 

    k             :: Int64 # number of analogs
    neighborhood  :: BitArray{2}
    catalog       :: Catalog
    regression    :: Symbol
    sampling      :: Symbol

    function AnalogForecasting( k       :: Int64, 
                                xt      :: TimeSeries, 
                                catalog :: Catalog;
                                regression = :local_linear,
                                sampling   = :gaussian )
    
        neighborhood = trues((xt.nv, xt.nv)) # global analogs
    
        new( k, neighborhood, catalog, regression, sampling)
    
    end 

    function AnalogForecasting( k            :: Int64, 
                                xt           :: TimeSeries, 
                                catalog      :: Catalog,
                                neighborhood :: BitArray{2},
                                regression   :: Symbol,
                                sampling     :: Symbol )
    
    
        new( k, neighborhood, catalog, regression, sampling)
    
    end 

end

""" 
    Apply the analog method on catalog of historical data 
    to generate forecasts. 
"""
function ( forecasting :: AnalogForecasting)(x :: Array{Float64,2})

    nv, np         = size(x)
    xf             = zeros(Float64, (nv,np))
    xf_mean        = zeros(Float64, (nv,np))
    ivar           = [1]
    condition      = true

    while condition

        if all(forecasting.neighborhood)
            ivar_neighboor = collect(1:nv)
            ivar           = collect(1:nv)
            condition      = false
        else
            ivar_neighboor = findall(vec(forecasting.neighborhood[ivar,:]))
        end

        # global analog forecasting
        kdt = KDTree( forecasting.catalog.analogs, leafsize=50)
        index_knn, dist_knn = knn(kdt, x, forecasting.k)
        
        # parameter of normalization for the kernels
        λ = median(Iterators.flatten(dist_knn))
        # compute weights
        weights = [exp.(-dist.^2 ./ λ) for dist in dist_knn]
        weights = [w ./ sum(w) for w in weights]

        # initialization
        xf_tmp = zeros(Float64, (last(ivar),forecasting.k))

        X = zeros(Float64,(length(ivar_neighboor), forecasting.k))
        Y = zeros(Float64,(length(ivar), forecasting.k))
        w = zeros(Float64, forecasting.k)

        for ip = 1:np
 
            if forecasting.regression == :locally_constant
                
                # compute the analog forecasts
                xf_tmp[ivar,:] .= forecasting.catalog.successors[ivar, index_knn[ip]]
                
                # weighted mean and covariance
                xf_mean[ivar,ip] = sum(xf_tmp[ivar,:] .* weights[ip]',dims=2)
                Exf   = xf_tmp[ivar,:] .- xf_mean[ivar,ip]
                cov_xf = Symmetric(1.0 ./(1.0 .- sum(weights[ip].^2)) 
                                   .* (Exf .* weights[ip]') * Exf')

            elseif forecasting.regression == :increment # method "locally-incremental"
                
                # compute the analog forecasts
                xf_tmp[ivar,:] .= (x[ivar,ip] 
                               .+ forecasting.catalog.successors[ivar,index_knn[ip]]
                               .- forecasting.catalog.analogs[ivar,index_knn[ip]])
                
                # weighted mean and covariance
                xf_mean[ivar,ip] = sum(xf_tmp[ivar,:] .* weights[ip]',dims=2)
                Exf = xf_tmp[ivar,:] .- xf_mean[ivar,ip]
                cov_xf = Symmetric(1.0 ./(1.0 .- sum(weights[ip].^2))
                                   .* (Exf .* weights[ip]') * Exf')

            elseif forecasting.regression == :local_linear

                # define analogs, successors and weights
                X .= forecasting.catalog.analogs[ ivar_neighboor , index_knn[ip]]
                Y .= forecasting.catalog.successors[ ivar, index_knn[ip]]
                w .= weights[ip]
                # compute centered weighted mean and weighted covariance
                Xm   = sum(X .* w', dims=2)
                Xc   = X .- Xm
                Xr   = vcat( ones(forecasting.k)', Xc)
                Cxx  = (Xr .* w') * Xr'
                Cxx2 = Symmetric((Xr .* w'.^2) * Xr')
                Cxy  = (Y  .* w') * Xr'
                inv_Cxx = pinv(Cxx, rtol=0.001) 
                # regression on principal components
                beta = Cxy * inv_Cxx 
                X0   = x[ivar_neighboor,ip] .- Xm
                X0r  = vcat([1], X0 )
                # weighted mean
                xf_mean[ivar,ip] = beta * X0r
                pred             = beta * Xr 
                res              = Y  .- pred
                xf_tmp[ivar,:]  .= xf_mean[ivar,ip] .+ res
                # weigthed covariance
                cov_xfc = Symmetric((res * (w .* res'))/(1 .- tr(Cxx2 * inv_Cxx)))
                cov_xf  = Symmetric(cov_xfc .* (1 .+ tr(Cxx2 * inv_Cxx * X0r * X0r' * inv_Cxx)))
                # constant weights for local linear
                weights[ip] .= 1.0/length(weights[ip])

            else

                @error "regression: locally_constant, :increment, :local_linear "

            end
            
            # Gaussian sampling
            if forecasting.sampling == :gaussian

                # random sampling from the multivariate Gaussian distribution
                d = MvNormal(xf_mean[ivar,ip], cov_xf)
                xf[ivar, ip] .= rand!(d, xf[ivar, ip])
            
            # Multinomial sampling
            elseif forecasting.sampling == :multinomial

                # random sampling from the multinomial distribution of the weights
                # this speedup is due to Peter Acklam
                cumprob = cumsum(weights[ip])
                R = rand()
                M = 1 :: Int64
                N = length(cumprob)
                for i in 1:N-1
                    M += R > cumprob[i]
                end
                igood = M
                xf[ivar, ip] .= xf_tmp[ivar, igood]
            
            else

                @error " sampling = :gaussian or :multinomial" 

            end
                
        end

        if all(ivar .== [nv]) || length(ivar) == nv
            condition = false
        else
            ivar .+= 1
        end

    end

    xf, xf_mean
    
end
