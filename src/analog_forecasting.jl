using NearestNeighbors

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
mutable struct AnalogForecasting <: AbstractForecasting 

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

end

""" 
    Apply the analog method on catalog of historical data 
    to generate forecasts. 
"""
function ( forecasting :: AnalogForecasting)(x :: Array{T,2}) where T

    nv, np         = size(x)
    xf             = zeros(T, (nv,np))
    xf_mean        = zeros(T, (nv,np))
    ivar           = [1]
    condition      = true

    while condition

        if all(forecasting.neighborhood)
            ivar_neighboor = collect(1:nv)
            ivar           = collect(1:nv)
            condition      = false
        else
            ivar_neighboor = findall(forecasting.neighborhood[ivar,:])
        end

        # global analog forecasting
        kdt = KDTree( forecasting.catalog.analogs, leafsize=50)
        index_knn, dist_knn = knn(kdt, x, forecasting.k)
        
        dists   = zeros(Float64,(forecasting.k, np))
        weights = zeros(Float64,(forecasting.k, np))
        dists  .= hcat(dist_knn...)
        # parameter of normalization for the kernels
        λ = median(dists)
        # compute weights
        weights .= exp.(-dists.^2 ./ λ)
        mk_stochastic!(weights)
        # initialization

        for ip = 1:np
 
            xf_tmp = zeros(Float64, (maximum(ivar),forecasting.k))

            if forecasting.regression == :locally_constant
                
                # compute the analog forecasts
                xf_tmp[i_var,:] = forecasting.catalog.successors[ivar, index_knn[ip]]
                
                # weighted mean and covariance
                xf_mean[i_var,ip] = sum(xf_tmp[ivar,:] .* weights[:,ip]',dims=2)
                E_xf   = xf_tmp[i_var,:] .- xf_mean[i_var,ip]
                cov_xf = Symmetric(1.0/(1.0 .- sum(weights[:,ip].^2)) .* (weights[:,ip] .* E_xf) * E_xf')


            elseif forecasting.regression == :local_linear

                # define analogs, successors and weights
                X = forecasting.catalog.analogs[ ivar_neighboor , index_knn[ip]]
                Y = forecasting.catalog.successors[ ivar, index_knn[ip]]
                w = weights[:,ip]
                # compute centered weighted mean and weighted covariance
                Xm   = sum(X .* w', dims=2)
                Xc   = X .- Xm
                # use SVD decomposition to compute principal components
                F    = svd(Xc')
                # keep eigen values higher than 1%
                ind  = findall(F.S ./ sum(F.S) .> 0.01) 
                Xr   = vcat( ones(1,size(Xc)[2]), F.Vt[ind,:] * Xc)
                Cxx  = Symmetric((Xr .* w') * Xr')
                Cxx2 = Symmetric((Xr .* w'.^2) * Xr')
                Cxy  = (Y  .* w') * Xr'
                inv_Cxx = inv(Cxx) # in case of error here, increase the number 
                                   # of analogs (k option)
                # regression on principal components
                beta = Cxy * inv_Cxx 
                X0   = x[ivar_neighboor,ip] .- Xm
                X0r  = vcat(ones(1,size(X0)[2]), F.Vt[ind,:] * X0 )
                # weighted mean
                xf_mean[ivar,ip] = beta * X0r
                pred             = beta * Xr 
                res              = Y  .- pred
                xf_tmp[ivar,:]  .= xf_mean[ivar,ip] .+ res
                # weigthed covariance
                cov_xfc = Symmetric((res * (w .* res'))/(1 .- tr(Cxx2 * inv_Cxx)))
                cov_xf  = Symmetric(cov_xfc .* (1 + tr(Cxx2 * inv_Cxx * X0r * X0r' * inv_Cxx)))
                # constant weights for local linear
                weights[:,ip] .= 1.0/length(weights[:,ip])
            end
            
            # random sampling from the multivariate Gaussian distribution
            d = MvNormal(xf_mean[ivar,ip], cov_xf)
            xf[ivar, ip] .= rand!(d, xf[ivar, ip])
                
        end

        if all(ivar .== [nv]) && length(ivar) == nv
            condition = false
        else
            ivar .+= 1
        end

    end

    xf, xf_mean
    
end
