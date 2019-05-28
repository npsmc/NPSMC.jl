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
function generate_data( ssm :: StateSpaceModel, 
                        u0  :: Vector{Float64},
                        seed = 42 )

    Random.seed!(seed)

    try @assert ssm.dt_states < ssm.dt_obs
    catch
       @error " ssm.dt_obs must be bigger than ssm.dt_states"
    end

    try @assert mod(ssm.dt_obs,ssm.dt_states) == 0.0
    catch
        @error " ssm.dt_obs must be a multiple of ssm.dt_states "
    end
    
    tspan = (0.0,ssm.nb_loop_test)
    prob  = ODEProblem(ssm.model, u0, tspan, ssm.params)
    
    # generSate true state (xt)
    sol = solve(prob,reltol=1e-6,saveat=ssm.dt_states*ssm.dt_integration)
    xt  = TimeSeries(sol.t, sol.u)
    
    # generate  partial/noisy observations (yo)
    nt   = xt.nt
    nv   = xt.nv
    d    = MvNormal(ssm.sigma2_obs .* Matrix(I,nv,nv))
    
    yo         = TimeSeries( xt.t, xt.u .* NaN)
    step       = ssm.dt_obs ÷ ssm.dt_states
    nt         = length(xt.t)
    ε          = rand(d, nt)
    
    for j in 1:step:nt
        for i in ssm.var_obs
            yo.u[j][i] = xt.u[j][i] + ε[i,j]
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

""" 
     generate_data(ssm, T_burnin, T; seed = 1)

Generate simulated data from Space State Model
"""
function generate_data(ssm :: SSM, x0 :: Vector{Float64}, nt :: Int64; seed = 1)

    Random.seed!(seed)

    nv = length(x0)
    μ = zeros(Float64, nv)
    d = MvNormal(μ, ssm.Q)
    # generate true state
    xt = [zeros(Float64, nv) for i in 1:nt]
    xt[1] .= x0
    dt = ssm.dt_model * ssm.dt_int 
    time = Float64[0.0]
    for t in 1:nt-1
        xtmp = xt[t]
        for i in 1:ssm.dt_model
            xtmp = ssm.mx(xtmp)
        end
        xt[t+1] .= xtmp .+ vec(rand(d, nv))
        push!(time, t * dt)
    end

    # generate  partial/noisy observations
    yo = xt .* NaN

    d = MvNormal(μ, ssm.R)
    for t in 1:nt-1
        yo[t] .= vec(ssm.h(xt[t+1]) .+ rand(d, nv))
    end


    TimeSeries(time, xt), TimeSeries(time[1:end-1], yo)

end
    
function train_test_split( X :: TimeSeries, Y :: TimeSeries; test_size=0.5)

    time    = X.t
    T       = length(time)
    T_test  = Int64(T * test_size)
    T_train = T - T_test

    X_train = TimeSeries( time[1:T_train], X.u[:, 1:T_train-1])
    Y_train = TimeSeries( time[2:end], Y.u[:, 1:T_train-1])

    X_test = TimeSeries(time,     X.u[:, T_train:end])
    Y_test = TimeSeries(time[2:end], Y.u[:, T_train:end-1])

    X_train, Y_train, X_test, Y_test

end
