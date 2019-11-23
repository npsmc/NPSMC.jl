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
    Cxy  :: Array{Float64, 2}
    pred :: Array{Float64, 2}
    X0r  :: Array{Float64, 2}

    function LocalLinear( k, ivar, ivar_neighboor)

        #X = zeros(Float64,(nvn, k))
        #Y = zeros(Float64,(nv, k))
        #w = zeros(Float64, k)

        nv  = length(ivar)
        nvn = length(ivar_neighboor)

        res  = zeros(Float64, (nv, k))
        beta = zeros(Float64, (nv, nvn+1))
        Xm   = zeros(Float64, (nvn, 1))
        Xc   = zeros(Float64, (nvn, k))
        Xr   = ones(Float64, (nvn+1, k))
        Cxx  = zeros(Float64, (nvn+1, nvn+1))
        Cxx2 = zeros(Float64, (nvn+1, nvn+1))
        Cxy  = zeros(Float64, (nv, nvn+1))
        pred = zeros(Float64, (nv, k))
        X0r  = ones(Float64, (nvn+1, 1))

        new(  k, ivar, ivar_neighboor, res, beta, Xm, Xc, Xr,
              Cxx, Cxx2, Cxy, pred, X0r)

    end

end 


function compute( ll :: LocalLinear, x, xf_tmp, xf_mean, ip, X, Y, w )

    ivar = ll.ivar
    ivar_neighboor = ll.ivar_neighboor

    ll.Xm  .= sum(X .* w', dims=2)
    ll.Xr[2:end,:] .= X .- ll.Xm
    mul!(ll.Cxx , (ll.Xr .* w'), ll.Xr')
    mul!(ll.Cxx2, (ll.Xr .* w'.^2), ll.Xr')
    mul!(ll.Cxy, (Y  .* w'), ll.Xr')
    ll.Cxx .= pinv(ll.Cxx, rtol=0.001) 
    # regression on principal components
    mul!(ll.beta, ll.Cxy, ll.Cxx) 
    ll.X0r[2:end,:] .= x[ivar_neighboor,ip] .- ll.Xm
    # weighted mean
    xf_mean[ivar,ip] = ll.beta * ll.X0r
    mul!(ll.pred, ll.beta, ll.Xr) 
    Y  .-= ll.pred
    xf_tmp[ivar,:]  .= xf_mean[ivar,ip] .+ Y
    # weigthed covariance
    cov_xfc = Symmetric((Y * (w .* Y'))/(1 .- tr(ll.Cxx2 * ll.Cxx)))
    cov_xf  = Symmetric(cov_xfc .* (1 .+ tr(ll.Cxx2 * ll.Cxx * ll.X0r * ll.X0r' * ll.Cxx)))

    return cov_xf

end

            
