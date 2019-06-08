@testset " Analog forecasting " begin

import NPSMC: normalise!, sample_discrete
import DifferentialEquations: ODEProblem, solve

n = 10
M = rand(n,3)

normalise!(M)

@test sum(M) ≈ 1.0

x  = range(0, stop=2π, length=10) |> collect
x .= sin.(x)
normalise!(x)
@show sample_discrete(x, 1, 1)

σ = 10.0
ρ = 28.0
β = 8.0/3

dt_integration = 0.01
dt_states      = 1 
dt_obs         = 8 
parameters     = [σ, ρ, β]
var_obs        = [1]
nb_loop_train  = 10^2 
nb_loop_test   = 10
sigma2_catalog = 0.0
sigma2_obs     = 2.0

ssm = StateSpaceModel( lorenz63,
                       dt_integration, dt_states, dt_obs, 
                       parameters, var_obs,
                       nb_loop_train, nb_loop_test,
                       sigma2_catalog, sigma2_obs )

# compute u0 to be in the attractor space
u0    = [8.0;0.0;30.0]
tspan = (0.0,5.0)
prob  = ODEProblem(ssm.model, u0, tspan, parameters)
u0    = last(solve(prob, reltol=1e-6, save_everystep=false))

xt, yo, catalog = generate_data( ssm, u0 )

for regression in [:locally_constant, :increment, :local_linear]
    for sampling in [:gaussian, :multinomial]
        f  = AnalogForecasting( 50, xt, catalog; 
                                regression = regression,
                                sampling   = sampling )
        for method in [EnKS(100), EnKF(100), PF(100)]
            data_assimilation = DataAssimilation( f, xt, ssm.sigma2_obs )
            x̂  = data_assimilation(yo, method)
            accuracy = RMSE(xt, x̂) 
            println( " $regression, $sampling, $method : $accuracy ")
            @test accuracy < 2.0
        end
    end
end


end
