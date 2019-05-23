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
        regression = :local_linear
        sampling   = :gaussian
    
        new( k, neighborhood, catalog, regression, sampling)
    
    end 

end 

""" 
    Apply the analog method on catalog of historical data 
    to generate forecasts. 
"""
function ( af :: AnalogForecasting)(x :: Array{T,2}) where T

    np, nv         = size(x)
    xf             = zeros(T, (np,nv))
    xf_mean        = zeros(T, (np,nv))
    ivar           = [1]
    stop_condition = false
    
    # local or global analog forecasting
    while !stop_condition

        if all(af.neighborhood) # in case of global approach
            ivar_neighboor = collect(1:nv)
            ivar           = collect(1:nv)
            stop_condition = true
        else                    # in case of local approach
            ivar_neighboor = af.neighborhood[ivar,:] 
        end
            
        # find the indices and distances of the k-nearest neighbors (knn)
        nc = size(af.catalog.analogs)
        data = zeros(Float64,(nc[2],nc[1]))
        for i in 1:nc[1], j in 1:nc[2]
            data[j,i] =  af.catalog.analogs[i,j]
        end

        kdt = KDTree(data, leafsize=50)

        nc = size(x)
        points = zeros(Float64,(nc[2],nc[1]))
        for i in 1:nc[1], j in 1:nc[2]
            points[j,i] =  x[i,j]
        end

        dist_knn, index_knn = knn(kdt, points, af.k)
        
        # parameter of normalization for the kernels
        λ = median(Iterators.flatten(dist_knn))

        # compute weights
        if af.k == 1
            weights = ones(Float64,(np,1))
        else
            weights = [exp(-d^2 / λ) for d in Iterators.flatten(dist_knn)]
            mk_stochastic!(weights)
        end

        @show weights

#        # for each member/particle
#        for ip in 1:np
#            
#            # initialization
#            xf_tmp = zeros((af.k,maximum(ivar)))
#         
#            # define analogs, successors and weights
#            X = af.catalog.analogs[    index_knn[ip,:], ivar_neighboor ]                
#            Y = af.catalog.successors[ index_knn[ip,:], ivar]
#
#            w = weights[ip,:]'
#            
#            # compute centered weighted mean and weighted covariance
#            Xm = sum(X .* w', dims=1)
#            @show Xc = X .- Xm
#
#            
#            # use SVD decomposition to compute principal components
#            F = svd(Xc)
#
#            ind = F.S ./ sum(F.S) .> 0.01 # keep eigen values higher than 1%
#
#            Xr   = vcat(ones(size(X)[0]), Xc * F.Vt[:,ind])
#            Cxx  = (w    .* Xr') * Xr
#            Cxx2 = (w.^2 .* Xr') * Xr
#            Cxy  = (w    .* Y' ) * Xr
#
#            inv_Cxx = inv(Cxx) # in case of error here, increase the number 
#                               # of analogs (af.k option)
#
#            
#            # regression on principal components
#            beta = inv_Cxx * Cxy'
#            X0   = x[ip,ivar_neighboor] .- Xm
#            X0r  = vcat(ones(size(X0)[0]), X0 * F.Vt[:,ind])
#
#            # weighted mean
#            xf_mean[ip,ivar] = X0r * beta
#            pred               = Xr  * beta
#            res                = Y  .- pred
#            xf_tmp[:,ivar]   .= xf_mean[ip,ivar] .+ res
#    
#            # weigthed covariance
#            cov_xfc = ((w .* res') * res)/(1 .- trace(Cxx2 * inv_Cxx))
#            cov_xf  = cov_xfc .* ( 1 .+ trace(Cxx2 * inv_Cxx * X0r' * X0r * inv_Cxx))
#            
#            # constant weights for local linear
#            weights[ip,:] .= 1.0/length(weights[ip,:])
#            
#            # random sampling from the multivariate Gaussian distribution
#            d = MvNormal(xf_mean[ip,ivar],cov_xf)
#            rand!(d, xf[ip,ivar])
#
#        end
            
        # stop condition
        if all(ivar .== nv) || length(ivar) == nv

            stop_condition = true
             
        else

            ivar .+= 1

        end

    end

    xf, xf_mean
    
end
