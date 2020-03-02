@testset "Lorenz 96" begin

using DifferentialEquations, Random, LinearAlgebra
using NPSMC

rng = MersenneTwister(123)
F = 8
J = 40 :: Int64
parameters = [F, J]
dt_integration = 0.05
dt_states = 1
dt_obs = 4 
var_obs = randperm(rng, J)[1:20]
nb_loop_train = 100 
nb_loop_test = 10 
sigma2_catalog = 0. 
sigma2_obs = 2. 

ssm = StateSpaceModel( lorenz96, 
                       dt_integration, dt_states, dt_obs, 
                       parameters, var_obs,
                       nb_loop_train, nb_loop_test,
                       sigma2_catalog, sigma2_obs )

# 5 time steps (to be in the attractor space)

u0 = F .* ones(Float64, J)
u0[J÷2] = u0[J÷2] + 0.01

tspan = (0.0,5.0)
p = [F, J]
prob  = ODEProblem(lorenz96, u0, tspan, p)
sol = solve(prob, reltol=1e-6, saveat= dt_integration)

u0 = last(sol.u)

xt, yo, catalog = generate_data(ssm, u0);

local_analog_matrix =  BitArray{2}(diagm( -2  => trues(xt.nv-2),
             -1  => trues(xt.nv-1),
              0  => trues(xt.nv),
              1  => trues(xt.nv-1),
              2  => trues(xt.nv-2),             
             J-2 => trues(xt.nv-(J-2)),
             J-1 => trues(xt.nv-(J-1)))
    + transpose(diagm( J-2 => trues(xt.nv-(J-2)),
             J-1 => trues(xt.nv-(J-1))))
    );

neighborhood = local_analog_matrix
regression = :local_linear
sampling   = :gaussian

@testset "Global analog with local linear regression " begin

    f  = AnalogForecasting( 100, xt, catalog, regression = regression, sampling = sampling)
    DA = DataAssimilation( f, xt, ssm.sigma2_obs )
    @time x̂ = forecast( DA, yo, EnKS(500), progress = true)
    rmse = RMSE(xt,x̂)
    println("RMSE(global analog  DA) = $rmse ")

    @test rmse < 5.0

end


@testset "Global analog with classic computation " begin

    DA = DataAssimilation( ssm, xt )
    @time x̂  = forecast(DA, yo, EnKS(500), progress = true);
    rmse = RMSE(xt,x̂)
    println("RMSE(global classic DA) = $rmse ")
    
    @test rmse < 1.0

end

@testset "Local analog with local linear regression " begin

    f  = AnalogForecasting( 100, xt, catalog, neighborhood, regression, sampling)
    DA = DataAssimilation( f, xt, ssm.sigma2_obs )
    @time x̂ = forecast( DA, yo, EnKS(500), progress = true)
    rmse = RMSE(xt,x̂)
    println("RMSE(local analog DA) = $rmse ")

    @test rmse < 5.0

end



@test true

end
