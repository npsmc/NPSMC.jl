struct LocalLinear

    k
    ivar
    ivar_neighboor

end 


function compute( local_linear :: LocalLinear, x, xf_tmp, xf_mean, ip, X, Y, w )

    ivar = local_linear.ivar
    ivar_neighboor = local_linear.ivar_neighboor

    # compute centered weighted mean and weighted covariance
    Xm   = sum(X .* w', dims=2)
    Xc   = X .- Xm
    Xr   = vcat( ones(local_linear.k)', Xc)
    Cxx  = (Xr .* w') * Xr'
    Cxx2 = Symmetric((Xr .* w'.^2) * Xr')
    Cxy  = (Y  .* w') * Xr'
    inv_Cxx = pinv(Cxx, rtol=0.01) 
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

    return cov_xf

end

            
