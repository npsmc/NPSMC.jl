using Statistics, NearestNeighbors

" Compute the Root Mean Square Error between 2 n-dimensional vectors. "
function RMSE(a, b)
 
    sqrt.(mean((a .- b).^2))

end

" Normalize the entries of a multidimensional array sum to 1. "
function normalise!(M)

    c = sum(M)
    # Set any zeros to one before dividing
    c += c == 0
    M ./= c

end

""" 
Ensure the matrix is stochastic, i.e., 
the sum over the last dimension is 1.
"""
function mk_stochastic!(T)

    if last(size(T)) == 1
        normalise!(T)
    else
        n = ndims(T)
        # Copy the normaliser plane for each i.
        normaliser = sum(T,dims=n)
        # Set zeros to 1 before dividing
        # This is valid since normaliser(i) = 0 iff T(i) = 0
        normaliser .+= normaliser .== 0
        T ./= normaliser
    end

end

""" 
Sampling from a non-uniform distribution. 
"""
function sample_discrete(prob)

    # this speedup is due to Peter Acklam
    cumprob = cumsum(prob)
    R = rand()
    M = 0 :: Int64
    N = length(cumprob)
    for i in 1:N-1
        M += R > cumprob[i]
    end
    M
end

export AnalogForecasting

"""
parameters of the analog forecasting method

- k : number of analogs
- neighborhood : global analogs
- catalog : catalog with analogs and successors
- regression : chosen regression ('locally_constant', 'increment', 'local_linear')
- sampling : chosen sampler ('gaussian', 'multinomial')
"""
struct AnalogForecasting

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
    while !stop_condition

        if all(af.neighborhood) # in case of global approach
            i_var_neighboor = collect(1:n)
            i_var           = collect(1:n)
            stop_condition  = true
        else                    # in case of local approach
            i_var_neighboor = af.neighborhood[i_var,:] 
        end
            
        # find the indices and distances of the k-nearest neighbors (knn)
        kdt = KDTree(AF.catalog.analogs[:,i_var_neighboor], 
                     leaf_size=50)

        dist_knn, index_knn = knn(kdt, x[:,i_var_neighboor], af.k)
        
        # parameter of normalization for the kernels
        lambdaa = median(dist_knn)

        # compute weights
        if af.k == 1
            weights = ones(Float64,(N,1))
        else
            weights = np.exp(-np.power(dist_knn,2)/lambdaa)
            mk_stochastic!(weights)
        end

#=
        
        # for each member/particle
        for i_N in range(0,N):
            
            # initialization
            xf_tmp = np.zeros([AF.k,np.max(i_var)+1])
         
            # define analogs, successors and weights
            X = AF.catalog.analogs[np.ix_(index_knn[i_N,:],i_var_neighboor)]                
            Y = AF.catalog.successors[np.ix_(index_knn[i_N,:],i_var)]                
            w = weights[i_N,:][np.newaxis]
            
            # compute centered weighted mean and weighted covariance
            Xm = np.sum(X*w.T, axis=0)[np.newaxis]
            Xc = X - Xm
            
            # use SVD decomposition to compute principal components
            U,S,V = np.linalg.svd(Xc,full_matrices=False)
            ind = np.nonzero(S/np.sum(S)>0.01)[0] # keep eigen values higher than 1%
            
            # regression on principal components
            Xr   = np.c_[np.ones(X.shape[0]), np.dot(Xc,V.T[:,ind])]
            Cxx  = np.dot(w    * Xr.T,Xr)
            Cxx2 = np.dot(w**2 * Xr.T,Xr)
            Cxy  = np.dot(w    * Y.T, Xr)
            inv_Cxx = inv(Cxx) # in case of error here, increase the number of analogs (AF.k option)
            beta = np.dot(inv_Cxx,Cxy.T)
            X0 = x[i_N,i_var_neighboor]-Xm
            X0r = np.c_[np.ones(X0.shape[0]),np.dot(X0,V.T[:,ind])]
             
            # weighted mean
            xf_mean[i_N,i_var] = np.dot(X0r,beta)
            pred = np.dot(Xr,beta)
            res = Y-pred
            xf_tmp[:,i_var] = xf_mean[i_N,i_var] + res
    
            # weigthed covariance
            cov_xfc = np.dot(w * res.T,res)/(1-np.trace(np.dot(Cxx2,inv_Cxx)))
            cov_xf = cov_xfc*(1+np.trace(Cxx2@inv_Cxx@X0r.T@X0r@inv_Cxx))
            
            # constant weights for local linear
            weights[i_N,:] = 1.0/len(weights[i_N,:])
            
            
            # random sampling from the multivariate Gaussian distribution
            xf[i_N,i_var] = np.random.multivariate_normal(xf_mean[i_N,i_var],cov_xf)
            

=#
        # stop condition
        if all(i_var .== n) && length(i_var) == n

            stop_condition = true
             
        else

            i_var .+= 1

        end

    end
            
    
end
