using Distributions
using LinearAlgebra
using ProgressMeter

abstract type MonteCarloMethod end

export DataAssimilation

"""
    DataAssimilation( forecasting, method, np, xt, sigma2) 

parameters of the filtering method
 - method :chosen method (:AnEnKF, :AnEnKS, :AnPF)
 - N      : number of members (AnEnKF/AnEnKS) or particles (AnPF)
"""
struct DataAssimilation

    xb     :: Vector{Float64}
    B      :: Array{Float64, 2}
    H      :: Array{Bool,    2}
    R      :: Array{Float64, 2}
    m      :: AbstractForecasting

    function DataAssimilation( m      :: AbstractForecasting,
                               xt     :: TimeSeries,
                               sigma2 :: Float64 )
                
        xb = xt.u[1]
        B  = 0.1 * Matrix(I, xt.nv, xt.nv)
        H  = Matrix( I, xt.nv, xt.nv)
        R  = sigma2 .* H

        new( xb, B, H, R, m )

    end

    function DataAssimilation( m      :: StateSpaceModel,
                               xt     :: TimeSeries )
                
        xb = xt.u[1]
        B  = 0.1 * Matrix(I, xt.nv, xt.nv)
        H  = Matrix( I, xt.nv, xt.nv)
        R  = m.sigma2_obs .* H

        new( xb, B, H, R, m )

    end

end
