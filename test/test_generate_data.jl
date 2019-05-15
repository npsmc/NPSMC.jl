@testset " Generate data for Lorenz-63 model " begin

    σ = 10.0
    ρ = 28.0
    β = 8.0/3

    dt_integration = 0.01
    dt_states      = 1 
    dt_obs         = 8 
    params         = [σ, ρ, β]
    var_obs        = [1]
    nb_loop_train  = 10^2 
    nb_loop_test   = 10
    sigma2_catalog = 0.0
    sigma2_obs     = 2.0

    ssm = StateSpace( dt_integration, dt_states, dt_obs, 
                      params, var_obs,
                      nb_loop_train, nb_loop_test,
                      sigma2_catalog, sigma2_obs )

    xt, yo, catalog = generate_data( ssm )

    @test true

end
