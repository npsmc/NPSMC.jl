export lorenz63, sinus

""" 

    lorenz63(du, u, p, t)

Lorenz-63 dynamical model 
```math
\\begin{eqnarray}
u̇₁(t) & = & p₁ ( u₂(t) - u₁(t)) \\\\
u̇₂(t) & = & u₁ ( p₂ - u₃(t)) - u₂(t) \\\\
u̇₃(t) & = & u₂(t)u₁(t) - p₃u₃(t)
\\end{eqnarray}
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
