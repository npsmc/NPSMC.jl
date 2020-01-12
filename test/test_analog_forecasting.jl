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

ssm = StateSpaceModel( lorenz63, dt_integration, dt_states, dt_obs, 
                       parameters, var_obs,
                       nb_loop_train, nb_loop_test, sigma2_catalog, sigma2_obs )

# compute u0 to be in the attractor space
u0    = [8.0;0.0;30.0]
tspan = (0.0,5.0)
prob  = ODEProblem(ssm.model, u0, tspan, parameters)
u0    = last(solve(prob, reltol=1e-6, save_everystep=false))

xt, yo, catalog = generate_data( ssm, u0 )

results = []
for regression in [:local_linear, :locally_constant, :increment]
    for sampling in [:gaussian, :multinomial]
        f  = AnalogForecasting( 50, xt, catalog; 
                                regression = regression,
                                sampling   = sampling )
        for method in [EnKS(100), EnKF(100), PF(100)]
            println(" $regression, $sampling, $method ")
            DA = DataAssimilation( f, xt, ssm.sigma2_obs )
            time = @elapsed x̂  = forecast(DA, yo, method)
            rmse = RMSE(xt, x̂) 
            push!(results, (regression, sampling, method, 
                            round(rmse,digits=3), round(time,digits=1)))
            @test rmse < 2.0
        end
    end
end

println()
for column in ["regression", "sampling", "method", "RMSE", "time"]
    print(rpad(column, 20, " "),)
end
println()
for res in sort(results, by=v -> v[4])

   for r in res 
      print(rpad(r, 20, " "))
   end
   println()

end

end
