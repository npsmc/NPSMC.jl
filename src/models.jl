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

"""
    lorenz96(S, t, F, J)

Lorenz-96 dynamical model 
"""
function Lorenz_96(dS, S, p, t)

    F, J = p
    x = zeros(Float64, J)
    x[1] = (S[2]-S[J-1])*S[J]-S[1]
    x[2] = (S[3]-S[J])*S[1]-S[2]
    x[J-1] = (S[1]-S[J-2])*S[J-1]-S[J]
    for j in 2:J-1
        x[j] = (S[j+1]-S[j-2])*S[j-1]-S[j]
    end
    dS = x' .+ F

end
