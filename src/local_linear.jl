struct LocalLinear

    k
    ivar
    ivar_neighboor
    res  :: Array{Float64, 2}
    beta :: Array{Float64, 2}
    Xm   :: Array{Float64, 2}
    Xc   :: Array{Float64, 2}
    Xr   :: Array{Float64, 2}
    Cxx  :: Array{Float64, 2}
    Cxx2 :: Array{Float64, 2}
    pred :: Array{Float64, 2}
    X0r  :: Array{Float64, 2}

    function LocalLinear( k, ivar, ivar_neighboor)

        #X = zeros(Float64,(nvn, k))
        #Y = zeros(Float64,(nv, k))
        #w = zeros(Float64, k)

        nv  = length(ivar)
        nvn = length(ivar_neighboor)

        res  = zeros(Float64, (nv, k))
        beta = zeros(Float64, (nvn+1, nvn+1))
        Xm   = zeros(Float64, (nvn, 1))
        Xc   = zeros(Float64, (nvn, k))
        Xr   = zeros(Float64, (nvn+1, k))
        Cxx  = zeros(Float64, (nvn+1, nvn+1))
        Cxx2 = zeros(Float64, (nvn+1, nvn+1))
        pred = zeros(Float64, (nvn, k))
        X0r  = zeros(Float64, (nvn+1, 1))

        new(  k, ivar, ivar_neighboor, res, beta, Xm, Xc, Xr,
              Cxx, Cxx2, pred, X0r)

    end

end 


function compute( ll :: LocalLinear, x, xf_tmp, xf_mean, ip, X, Y, w )

    ivar = ll.ivar
    ivar_neighboor = ll.ivar_neighboor

    # compute centered weighted mean and weighted covariance
    ll.Xm   .= sum(X .* w', dims=2)
    ll.Xc   .= X .- ll.Xm
    ll.Xr   .= vcat( ones(ll.k)', ll.Xc)
    ll.Cxx  .= (ll.Xr .* w') * ll.Xr'
    ll.Cxx2 .= (ll.Xr .* w'.^2) * ll.Xr'
    Cxy     = (Y  .* w') * ll.Xr'
    inv_Cxx = pinv(ll.Cxx, rtol=0.01) 
    # regression on principal components
    beta = Cxy * inv_Cxx 
    
    X0   = x[ivar_neighboor,ip] .- ll.Xm
    ll.X0r .= vcat([1], X0 )
    # weighted mean
    xf_mean[ivar,ip] = beta * ll.X0r
    mul!(ll.pred, beta, ll.Xr )
    ll.res          .= Y  .- ll.pred
    xf_tmp[ivar,:]  .= xf_mean[ivar,ip] .+ ll.res
    # weigthed covariance
    cov_xfc = Symmetric((ll.res * (w .* ll.res'))/(1 .- tr(ll.Cxx2 * inv_Cxx)))
    cov_xf  = Symmetric(cov_xfc .* (1 .+ tr(ll.Cxx2 * inv_Cxx * ll.X0r * ll.X0r' * inv_Cxx)))

    return cov_xf

end

            
