@testset " Generate data for Lorenz-63 model " begin

    import Random
    using Distributions
    using LinearAlgebra

    Random.seed!(1)

    σ = 10.0
    ρ = 28.0
    β = 8.0/3

    dt_states      = 1 
    dt_obs         = 8 
    params         = [σ, ρ, β]
    var_obs        = [0]
    nb_loop_train  = 10^2 
    nb_loop_test   = 10
    sigma2_catalog = 0.0
    sigma2_obs     = 2.0

    ssm = StateSpace( dt_states, dt_obs, params, var_obs,
                      nb_loop_train, nb_loop_test,
                      sigma2_catalog, sigma2_obs )


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
    sol = solve(prob,reltol=1e-6,saveat=dt_states)
    xt  = TimeSeries(sol.t, sol.u)

    # generate  partial/noisy observations (yo)
    d   = MvNormal(ssm.sigma2_obs*Matrix(I,3,3))

    n   = length(xt.values)
    eps = rand(d, n)

    @show eps'
    #yo_tmp = xt.values .+ eps
    #t_yo   = collect(0:ssm.dt_obs:last(xt.time))
#    i_t_obs = np.nonzero(np.in1d(t_xt,t_yo))[0]
#    yo.values = xt.values*np.nan
#    yo.values[np.ix_(i_t_obs,ssm.var_obs)] = yo_tmp[np.ix_(i_t_obs,ssm.var_obs)]
#    yo.time = xt.time
#    
#    #generate catalog
#    S =  odeint(Lorenz_63,S[S.shape[0]-1,:],
#                np.arange(0.01,ssm.nb_loop_train+0.000001,ssm.dt_integration),
#                args=(ssm.parameters.sigma,ssm.parameters.rho,ssm.parameters.beta))
#    T_train = S.shape[0]
#    eta = np.random.multivariate_normal(np.zeros(3),ssm.sigma2_catalog*np.eye(3,3),T_train)
#    catalog_tmp = S+eta
#    catalog.analogs = catalog_tmp[0:-ssm.dt_states,:]
#    catalog.successors = catalog_tmp[ssm.dt_states:,:]
#    catalog.source = ssm.parameters
    
    @test true

end
