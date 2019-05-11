using Random, LinearAlgebra

export TimeSeries

mutable struct TimeSeries

   time   :: Vector{Real}
   values :: Array{Union{Real,Missing},2}

end

export StateSpace

struct StateSpace

    dt_integration :: Real
    dt_states      :: Int
    dt_obs         :: Int
    params         :: Vector{Real}
    var_obs        :: Vector{Int64}
    nb_loop_train  :: Int
    nb_loop_test   :: Int
    sigma2_catalog :: Real
    sigma2_obs     :: Real
    
end

export Catalog

mutable struct Catalog

   analogs    :: Array{Real,2}
   successors :: Array{Real,2}
   sources    :: StateSpace

end

export generate_data

function generate_data( ssm )

    @assert ssm.dt_states < ssm.dt_obs
    # @error " ssm.dt_obs must be bigger than ssm.dt_states"
    @assert mod(ssm.dt_obs,ssm.dt_states) == 0.0
    # @error " ssm.dt_obs must be a multiple of ssm.dt_states "
    
    # 5 time steps (to be in the attractor space)       
    x0 = [8.0;0.0;30.0]
    tspan = (0.0,5.0)
    p = [10.0,28.0,8/3]
    prob = ODEProblem(lorenz63, x0, tspan, p)
    
    x0 = last(solve(prob, reltol=1e-6, save_everystep=false))
    tspan = (0.0,ssm.nb_loop_test)
    prob = ODEProblem(lorenz63, x0, tspan, p)
    
    # generSate true state (xt)
    sol = solve(prob,reltol=1e-6,saveat=ssm.dt_integration)
    xt  = TimeSeries(sol.t, vcat(sol.u'...))
    
    # generate  partial/noisy observations (yo)
    d       = MvNormal(ssm.sigma2_obs .* Matrix(I,3,3))
    nvalues = length(xt.time)
    eps     = rand(d, nvalues)
    
    yo   = TimeSeries( xt.time, xt.values .* NaN)
    step = ssm.dt_obs ÷ ssm.dt_states
    n    = length(xt.time)
    for j in ssm.var_obs
        for i in 1:step:n
            yo.values[i,j] = xt.values[i,j] + eps[j,i]
        end
    end
    
    #generate catalog
    x0 = last(sol)
    tspan = (0.0,ssm.nb_loop_train)
    prob = ODEProblem(lorenz63, x0, tspan, p)
    sol = solve(prob,reltol=1e-6,saveat= ssm.dt_integration)
    n = length(sol.t)
    catalog_tmp = vcat(sol.u'...) 
    if ssm.sigma2_catalog > 0
        d   = MvNormal([0.,0.0,.0], ssm.sigma2_catalog .* Matrix(I,3,3))
        eta = rand(d, n)
        catalog_tmp .+= eta'
    end

    Catalog( catalog_tmp[1:end-ssm.dt_states,:],
             catalog_tmp[ssm.dt_states:end,:], 
             ssm )

end

