using NearestNeighbors

export AnalogForecasting

"""
parameters of the analog forecasting method

- k : number of analogs
- neighborhood : global analogs
- catalog : catalog with analogs and successors
- regression : chosen regression ('locally_constant', 'increment', 'local_linear')
- sampling : chosen sampler ('gaussian', 'multinomial')
"""
mutable struct AnalogForecasting <: AbstractForecasting 

    k             :: Int64 # number of analogs
    neighborhood  :: BitArray{2}
    catalog       :: Catalog
    regression    :: Symbol
    sampling      :: Symbol


    function AnalogForecasting( k       :: Int64, 
                                xt      :: TimeSeries, 
                                catalog :: Catalog)
    
        neighborhood = trues((xt.nv, xt.nv)) # global analogs
        regression   = :local_linear
        sampling     = :gaussian
    
        new( k, neighborhood, catalog, regression, sampling)
    
    end 

end 

""" 
    Apply the analog method on catalog of historical data 
    to generate forecasts. 
"""
function ( af :: AnalogForecasting)(x :: Array{T,2}) where T

    nv, np         = size(x)
    xf             = zeros(T, (nv,np))
    xf_mean        = zeros(T, (nv,np))
    ivar_neighboor = 1:nv
    ivar           = 1:nv
    # global analog forecasting
    kdt = KDTree( af.catalog.analogs, leafsize=50)
    index_knn, dist_knn = knn(kdt, x, af.k)
    
    dists   = zeros(Float64,(af.k,np))
    weights = zeros(Float64,(af.k,np))
    dists  .= hcat(dist_knn...)
    # parameter of normalization for the kernels
    λ = median(dists)
    # compute weights
    weights .= exp.(-dists.^2 ./ λ)
    mk_stochastic!(weights)
    # initialization
    xf_tmp = zeros((maximum(ivar),af.k))

    for ip = 1:np
 
        # define analogs, successors and weights
        X = af.catalog.analogs[    ivar_neighboor , index_knn[ip]]
        Y = af.catalog.successors[ ivar, index_knn[ip]]
        w = weights[:,ip]
        # compute centered weighted mean and weighted covariance
        Xm = sum(X .* w', dims=2)
        Xc = X .- Xm
        # use SVD decomposition to compute principal components
        F = svd(Xc', full=true)
        # keep eigen values higher than 1%
        ind = findall(F.S ./ sum(F.S) .> 0.01) 
        Xr = vcat( ones(1,size(Xc)[2]), F.Vt[ind,:] * Xc)
        Cxx  = Symmetric((Xr .* w') * Xr')
        Cxx2 = Symmetric((Xr .* w'.^2) * Xr')
        Cxy  = (Y  .* w') * Xr'
        inv_Cxx = inv(Cxx) # in case of error here, increase the number 
                           # of analogs (af.k option)
        # regression on principal components
        beta =  Cxy * inv_Cxx 
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
        
        # random sampling from the multivariate Gaussian distribution
        @assert ishermitian(cov_xf)
        d = MvNormal(xf_mean[ivar,ip], cov_xf)
        xf[ivar, ip] .= rand!(d, xf[ivar, ip])
            
    end
            
    xf, xf_mean
    
end
