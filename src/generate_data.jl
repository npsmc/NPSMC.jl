using Random, LinearAlgebra

import DifferentialEquations: SDEProblem, solve
import Distributions: MvNormal, rand

export generate_data

"""
    generate_data( ssm, u0; seed=42)

from StateSpace generate:
 - true state (xt)
 - partial/noisy observations (yo)
 - catalog
"""
function generate_data(ssm::StateSpaceModel, u0::Vector{Float64}, seed = 42)

    rng = MersenneTwister(seed)

    try
        @assert ssm.dt_states < ssm.dt_obs
    catch
        @error " ssm.dt_obs must be bigger than ssm.dt_states"
    end

    try
        @assert mod(ssm.dt_obs, ssm.dt_states) == 0.0
    catch
        @error " ssm.dt_obs must be a multiple of ssm.dt_states "
    end

    tspan = (0.0, ssm.nb_loop_test)

    function σ( du, u, p, t)

        for i in eachindex(du)
            du[i] = ssm.sigma2_obs
        end

    end

    prob = SDEProblem(ssm.model, σ, u0, tspan, ssm.params)

    # generSate true state (xt)
    sol = solve(prob, saveat = ssm.dt_states * ssm.dt_integration)
    xt = TimeSeries(sol.t, sol.u)

    # generate  partial/noisy observations (yo)
    nt = xt.nt
    nv = xt.nv

    yo = TimeSeries(xt.t, xt.u .* NaN)
    step = ssm.dt_obs ÷ ssm.dt_states
    nt = length(xt.t)

    d = MvNormal(ssm.sigma2_obs .* Matrix(I, nv, nv))
    ε = rand(d, nt)
    for j = 1:step:nt
        for i in ssm.var_obs
            yo.u[j][i] = xt.u[j][i] + ε[i, j]
        end
    end

    # generate catalog
    u0 = last(sol)
    tspan = (0.0, ssm.nb_loop_train)
    prob = SDEProblem(ssm.model, σ, u0, tspan, ssm.params)
    sol = solve(prob, saveat = ssm.dt_integration)
    n = length(sol.t)

    catalog_tmp = sol.u

    xt, yo, Catalog(hcat(catalog_tmp...), ssm)

end

""" 
     generate_data(ssm, T_burnin, T; seed = 1)

Generate simulated data from Space State Model
"""
function generate_data(ssm::SSM, x0::Vector{Float64}, nt::Int64; seed = 1)

    Random.seed!(seed)

    nv = length(x0)
    μ = zeros(Float64, nv)
    d = MvNormal(μ, ssm.Q)
    # generate true state
    xt = [zeros(Float64, nv) for i = 1:nt]
    xt[1] .= x0
    dt = ssm.dt_model * ssm.dt_int
    time = Float64[0.0]
    for t = 1:nt-1
        xtmp = xt[t]
        for i = 1:ssm.dt_model
            xtmp = ssm.mx(xtmp)
        end
        xt[t+1] .= xtmp .+ vec(rand(d, nv))
        push!(time, t * dt)
    end

    # generate  partial/noisy observations
    yo = xt .* NaN

    d = MvNormal(μ, ssm.R)
    for t = 1:nt-1
        yo[t] .= vec(ssm.h(xt[t+1]) .+ rand(d, nv))
    end


    TimeSeries(time, xt), TimeSeries(time[1:end-1], yo)

end

"""
    train_test_split( X, Y; test_size)

Split time series into random train and test subsets
"""
function train_test_split(X::TimeSeries, Y::TimeSeries; test_size = 0.5)

    time = X.t
    T = length(time)
    T_test = Int64(T * test_size)
    T_train = T - T_test

    X_train = TimeSeries(time[1:T_train], X.u[:, 1:T_train-1])
    Y_train = TimeSeries(time[2:end], Y.u[:, 1:T_train-1])

    X_test = TimeSeries(time, X.u[:, T_train:end])
    Y_test = TimeSeries(time[2:end], Y.u[:, T_train:end-1])

    X_train, Y_train, X_test, Y_test

end
