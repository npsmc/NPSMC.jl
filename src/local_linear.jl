struct LocalLinear

    k
    ivar
    ivar_neighboor
    res  :: Array{Float64, 2}
    beta :: Array{Float64, 2}
    Xm   :: Array{Float64, 2}
    Xc   :: Array{Float64, 2}

    function LocalLinear( k, ivar, ivar_neighboor)

        #X = zeros(Float64,(length(ivar_neighboor), forecasting.k))
        #Y = zeros(Float64,(length(ivar), forecasting.k))
        #w = zeros(Float64, forecasting.k)

        nv  = length(ivar)
        nvn = length(ivar_neighboor)

        res  = zeros(Float64, (nv, k))
        beta = zeros(Float64, (nvn+1, nvn+1))
        Xm   = zeros(Float64, (nvn, 1))
        Xc   = zeros(Float64, (nvn, k))

        new(  k, ivar, ivar_neighboor, res, beta, Xm, Xc)

    end

end 


function compute( ll :: LocalLinear, x, xf_tmp, xf_mean, ip, X, Y, w )

    ivar = ll.ivar
    ivar_neighboor = ll.ivar_neighboor

    # compute centered weighted mean and weighted covariance
    ll.Xm  .= sum(X .* w', dims=2)
    ll.Xc  .= X .- ll.Xm
    Xr      = vcat( ones(ll.k)', ll.Xc)
    Cxx     = (Xr .* w') * Xr'
    Cxx2    = (Xr .* w'.^2) * Xr'
    Cxy     = (Y  .* w') * Xr'
    inv_Cxx = pinv(Cxx, rtol=0.01) 
    # regression on principal components
    beta = Cxy * inv_Cxx 
    
    X0   = x[ivar_neighboor,ip] .- ll.Xm
    X0r  = vcat([1], X0 )
    # weighted mean
    xf_mean[ivar,ip] = beta * X0r
    pred             = beta * Xr 
    ll.res          .= Y  .- pred
    xf_tmp[ivar,:]  .= xf_mean[ivar,ip] .+ ll.res
    # weigthed covariance
    cov_xfc = Symmetric((ll.res * (w .* ll.res'))/(1 .- tr(Cxx2 * inv_Cxx)))
    cov_xf  = Symmetric(cov_xfc .* (1 .+ tr(Cxx2 * inv_Cxx * X0r * X0r' * inv_Cxx)))

    return cov_xf

end

            
