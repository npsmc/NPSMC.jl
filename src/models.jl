export lorenz63, lorenz96, sinus

""" 

    lorenz63(du, u, p, t)

Lorenz-63 dynamical model 
```math
u̇₁(t) & = & p₁ ( u₂(t) - u₁(t)) \\\\
u̇₂(t) & = & u₁ ( p₂ - u₃(t)) - u₂(t) \\\\
u̇₃(t) & = & u₂(t)u₁(t) - p₃u₃(t)
```
"""
function lorenz63(du, u, p, t)

    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[2] - u[3]) - u[2]
    du[3] = u[1] *  u[2] - p[3]  * u[3]

end

""" 
    sinus(du, u, p, t)

Sinus toy dynamical model 
```math
u̇₁ = p₁ \\cos(p₁t) 
```
"""
function sinus(du, u, p, t)

    du[1] = p[1] * cos( p[1] * t )

end

"""
    lorenz96(S, t, F, J)

Lorenz-96 dynamical model. For ``i=1,...,N``:

```math
\\frac{dx_i}{dt} = (x_{i+1}-x_{i-2})x_{i-1} - x_i + F
```

where it is assumed that ``x_{-1}=x_{N-1},x_0=x_N`` and ``x_{N+1}=x_1``. 
Here ``x_i`` is the state of the system and ``F`` is a forcing constant. 

"""
function lorenz96(dx, x, p, t ) 
    F = p[1]
    N = Int64(p[2])
    # 3 edge cases
    dx[1] = (x[2] - x[N - 1]) * x[N] - x[1] + F
    dx[2] = (x[3] - x[N]) * x[1] - x[2] + F
    dx[N] = (x[1] - x[N - 2]) * x[N - 1] - x[N] + F
    # then the general case
    for n in 3:(N - 1)
      dx[n] = (x[n + 1] - x[n - 2]) * x[n - 1] - x[n] + F
    end
end
