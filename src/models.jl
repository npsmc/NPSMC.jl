export lorenz63

""" 

Lorenz-63 dynamical model 
```julia
u0 = [1.0,0.0,0.0]
tspan = (0.0,1.0)
p = [10.0,28.0,8/3]
prob = ODEProblem(lorenz63, u0, tspan, p)
```
"""
function lorenz63(du, u, p, t)

    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[2] - u[3]) - u[2]
    du[3] = u[1] *  u[2] - p[3]  * u[3]

end
