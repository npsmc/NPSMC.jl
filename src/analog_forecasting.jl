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
struct AnalogForecasting <: AbstractForecasting

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

    N, n           = size(x)
    xf             = zeros(T, (N,n))
    xf_mean        = zeros(T, (N,n))
    stop_condition = false
    i_var          = [1]
    
    # local or global analog forecasting
    #while !stop_condition

        if all(af.neighborhood) # in case of global approach
            i_var_neighboor = collect(1:n)
            i_var           = collect(1:n)
            stop_condition  = true
        else                    # in case of local approach
            i_var_neighboor = af.neighborhood[i_var,:] 
        end
            
        # find the indices and distances of the k-nearest neighbors (knn)
        kdt = KDTree(af.catalog.analogs[:,i_var_neighboor], 
                     leafsize=50)

        dist_knn, index_knn = knn(kdt, x[:,i_var_neighboor], af.k)
        
        # parameter of normalization for the kernels
        @show lambdaa = median(dist_knn)
        exit(0)

        # compute weights
        if af.k == 1
            weights = ones(Float64,(N,1))
        else
            weights = np.exp(-np.power(dist_knn,2)/lambdaa)
            mk_stochastic!(weights)
        end

        # for each member/particle
        for i_N in 1:N
            
            # initialization
            xf_tmp = zeros((af.k,maximum(i_var)))
         
            # define analogs, successors and weights
            X = [af.catalog.analogs[i,j] 
                    for i in index_knn[i_N,:], j in i_var_neighboor ]                
            Y = [af.catalog.successors[i,j] 
                    for i in index_knn[i_N,:], j in i_var]                

            w = weights[i_N,:]'
            
            # compute centered weighted mean and weighted covariance
            Xm = sum(X .* w', dims=1)
            Xc = X - Xm
            
            # use SVD decomposition to compute principal components
            F = svd(Xc)

            ind = F.S ./ sum(F.S) .> 0.01 # keep eigen values higher than 1%

            Xr   = vcat(ones(size(X)[0]), Xc * F.Vt[:,ind])
            Cxx  = (w    .* Xr') * Xr
            Cxx2 = (w.^2 .* Xr') * Xr
            Cxy  = (w    .* Y' ) * Xr

            inv_Cxx = inv(Cxx) # in case of error here, increase the number 
                               # of analogs (af.k option)

            
            # regression on principal components
            beta = inv_Cxx * Cxy'
            X0   = x[i_N,i_var_neighboor] .- Xm
            X0r  = vcat(ones(size(X0)[0]), X0 * F.Vt[:,ind])

            # weighted mean
            xf_mean[i_N,i_var] = X0r * beta
            pred               = Xr  * beta
            res                = Y  .- pred
            xf_tmp[:,i_var]   .= xf_mean[i_N,i_var] .+ res
    
            # weigthed covariance
            cov_xfc = ((w .* res') * res)/(1 .- trace(Cxx2 * inv_Cxx))
            cov_xf  = cov_xfc .* ( 1 .+ trace(Cxx2 * inv_Cxx * X0r' * X0r * inv_Cxx))
            
            # constant weights for local linear
            weights[i_N,:] .= 1.0/length(weights[i_N,:])
            
            # random sampling from the multivariate Gaussian distribution
            d = MvNormal(xf_mean[i_N,i_var],cov_xf)
            rand!(d, xf[i_N,i_var])

        #end
            
        # stop condition
        if all(i_var .== n) && length(i_var) == n

            stop_condition = true
             
        else

            i_var .+= 1

        end

    end
    
end
