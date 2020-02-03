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

    kdt::KDTree
    k::Int64 # number of analogs
    neighborhood::BitArray{2}
    catalog::Catalog
    regression::Symbol
    sampling::Symbol

    function AnalogForecasting(
        k::Int64,
        xt::TimeSeries,
        catalog::Catalog;
        regression = :local_linear,
        sampling = :gaussian,
        leafsize = 50,
    )

        neighborhood = trues((xt.nv, xt.nv)) # global analogs

        kdt = KDTree(catalog.analogs, leafsize = leafsize)

        new(kdt, k, neighborhood, catalog, regression, sampling)

    end

    function AnalogForecasting(
        k::Int64,
        xt::TimeSeries,
        catalog::Catalog,
        neighborhood::BitArray{2},
        regression::Symbol,
        sampling::Symbol,
        leafsize = 50,
    )

        kdt = KDTree(catalog.analogs, leafsize = leafsize)

        new(kdt, k, neighborhood, catalog, regression, sampling)

    end

end

""" 
    Apply the analog method on catalog of historical data 
    to generate forecasts. 
"""
function (forecasting::AnalogForecasting)(x::Array{Float64,2})

    nv, np = size(x)
    xf = zeros(Float64, (nv, np))
    xf_mean = zeros(Float64, (nv, np))
    ivar = [1]
    condition = true

    # global analog forecasting
    index_knn, dist_knn = knn(forecasting.kdt, x, forecasting.k)
    # parameter of normalization for the kernels
    λ = median(Iterators.flatten(dist_knn))

    while condition

        if all(forecasting.neighborhood)
            ivar_neighboor = 1:nv
            ivar = 1:nv
            condition = false
        else
            ivar_neighboor = findall(vec(forecasting.neighborhood[ivar, :]))
        end

        # compute weights
        weights = [exp.(-dist .^ 2 ./ λ) for dist in dist_knn]
        for (ip, w) in enumerate(weights)
            weights[ip] .= w ./ sum(w)
        end

        # initialization
        xf_tmp = zeros(Float64, (last(ivar), forecasting.k))

        if forecasting.regression == :local_linear
            local_linear = LocalLinear(forecasting.k, ivar, ivar_neighboor)
        end

        for ip = 1:np
            if forecasting.regression == :locally_constant

                # compute the analog forecasts
                xf_tmp[ivar, :] .= forecasting.catalog.successors[ivar, index_knn[ip]]
                # weighted mean and covariance
                xf_mean[ivar, ip] = sum(xf_tmp[ivar, :] .* weights[ip]', dims = 2)
                Exf = xf_tmp[ivar, :] .- xf_mean[ivar, ip]
                cov_xf = Symmetric(1.0 ./ (1.0 .- sum(weights[ip] .^ 2)) .*
                                   (Exf .* weights[ip]') * Exf')

            elseif forecasting.regression == :increment # method "locally-incremental"
                # compute the analog forecasts
                xf_tmp[ivar, :] .= (x[ivar, ip] .+
                                    forecasting.catalog.successors[ivar, index_knn[ip]] .-
                                    forecasting.catalog.analogs[ivar, index_knn[ip]])

                # weighted mean and covariance
                xf_mean[ivar, ip] = sum(xf_tmp[ivar, :] .* weights[ip]', dims = 2)
                Exf = xf_tmp[ivar, :] .- xf_mean[ivar, ip]
                cov_xf = Symmetric(1.0 ./ (1.0 .- sum(weights[ip] .^ 2)) .*
                                   (Exf .* weights[ip]') * Exf')

            elseif forecasting.regression == :local_linear

                # define analogs, successors and weights

                # compute centered weighted mean and weighted covariance
                cov_xf = compute(
                    local_linear,
                    x,
                    xf_tmp,
                    xf_mean,
                    ip,
                    forecasting.catalog.analogs[ivar_neighboor, index_knn[ip]],
                    forecasting.catalog.successors[ivar, index_knn[ip]],
                    weights[ip],
                )

                if cov_xf == 0 # error in pinv back to locally constant

                    xf_tmp[ivar, :] .= forecasting.catalog.successors[ivar, index_knn[ip]]
                    # weighted mean and covariance
                    xf_mean[ivar, ip] = sum(xf_tmp[ivar, :] .* weights[ip]', dims = 2)
                    Exf = xf_tmp[ivar, :] .- xf_mean[ivar, ip]
                    cov_xf = Symmetric(1.0 ./ (1.0 .- sum(weights[ip] .^ 2)) .*
                                       (Exf .* weights[ip]') * Exf')
                else

                    # constant weights for local linear
                    weights[ip] .= 1.0 / forecasting.k
                end

            else

                @error "regression: locally_constant, :increment, :local_linear "
            end

            # Gaussian sampling
            if forecasting.sampling == :gaussian

                if !isposdef(cov_xf)
                    cov_xf = PDMat(ensure_pos_sym(cov_xf))
                end

                # random sampling from the multivariate Gaussian distribution
                d = MvNormal(xf_mean[ivar, ip], cov_xf)
                xf[ivar, ip] .= rand!(d, xf[ivar, ip])

            # Multinomial sampling
            elseif forecasting.sampling == :multinomial

                # random sampling from the multinomial distribution 
                # of the weights
                # this speedup is due to Peter Acklam
                cumprob = cumsum(weights[ip])
                R = rand()
                M = 1::Int64
                N = length(cumprob)
                for i = 1:N-1
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
