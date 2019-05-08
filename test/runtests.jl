using Test

using DifferentialEquations

@testset "Lorenz 63" begin

function g(du,u,p,t)
 du[1] = p[1]*(u[2]-u[1])
 du[2] = u[1]*(p[2]-u[3]) - u[2]
 du[3] = u[1]*u[2] - p[3]*u[3]
end
u0 = [1.0;0.0;0.0]
tspan = (0.0,30.0)
p = [10.0,28.0,8/3]
prob = ODEProblem(g,u0,tspan,p)

@test true

end

