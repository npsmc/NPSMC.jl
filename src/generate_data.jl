using Random, LinearAlgebra

import DifferentialEquations: ODEProblem, solve
import Distributions: MvNormal, rand

export generate_data

"""
from StateSpace generate:
 - true state (xt)
 - partial/noisy observations (yo)
 - catalog
"""
function generate_data( ssm :: StateSpaceModel, u0 :: Vector{Float64} )

    @assert ssm.dt_states < ssm.dt_obs
    # @error " ssm.dt_obs must be bigger than ssm.dt_states"
    @assert mod(ssm.dt_obs,ssm.dt_states) == 0.0
    # @error " ssm.dt_obs must be a multiple of ssm.dt_states "
    
    tspan = (0.0,ssm.nb_loop_test)
    prob  = ODEProblem(ssm.model, u0, tspan, ssm.params)
    
    # generSate true state (xt)
    sol = solve(prob,reltol=1e-6,saveat=ssm.dt_states*ssm.dt_integration)
    xt  = TimeSeries(sol.t, sol.u)
    
    # generate  partial/noisy observations (yo)
    nt   = xt.nt
    nv   = xt.nv
    d    = MvNormal(ssm.sigma2_obs .* Matrix(I,nv,nv))
    
    yo         = TimeSeries( xt.time, xt.values .* NaN)
    step       = ssm.dt_obs ÷ ssm.dt_states
    nt         = length(xt.time)
    ε          = rand(d, nt)
    
    for j in 1:step:nt
        for i in ssm.var_obs
            yo.values[j][i] = xt.values[j][i] + ε[i,j]
        end
    end
    
    # generate catalog
    u0    = last(sol)
    tspan = (0.0,ssm.nb_loop_train)
    prob  = ODEProblem(ssm.model, u0, tspan, ssm.params)
    sol   = solve(prob,reltol=1e-6,saveat=ssm.dt_integration)
    n     = length(sol.t)
    @show size(sol.u)
    if ssm.sigma2_catalog > 0
        μ   = zeros(Float64,nv)
        σ   = ssm.sigma2_catalog .* Matrix(I,nv,nv)
        d   = MvNormal(μ, σ)
        η   = rand(d, n)
        catalog_tmp = [ η[:,j] .+ u for u in sol.u]
    else
        catalog_tmp = sol.u
    end

    xt, yo, Catalog( hcat(catalog_tmp...), ssm )

end
