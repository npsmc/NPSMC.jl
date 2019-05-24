import DifferentialEquations: ODEProblem, MonteCarloProblem, solve

abstract type AbstractForecasting end

export ModelForecasting

struct ModelForecasting <: AbstractForecasting

    dt     :: Float64
    params :: Vector{Float64}
    model  :: Function

    function ModelForecasting( ssm :: StateSpaceModel )

        new( ssm.dt_integration, ssm.params, ssm.model )

    end


end

""" 
    Apply the dynamical models to generate numerical forecasts. 
"""
function ( m :: ModelForecasting )( x :: Array{Float64, 2})

    np, nv = size(x)
    p      = m.params
    tspan  = (0.0, m.dt)
    u0     = zeros(Float64,nv)

    prob   = ODEProblem( ssm.model, u0, tspan, p)

    function prob_func( prob, i, repeat)
        prob.u0 .= x[:, i]
        prob
    end

    monte_prob = MonteCarloProblem(prob, prob_func=prob_func)

    sim = solve(monte_prob, Tsit5(), num_monte=np, save_everystep=false)

    sol = [last(sim[i].u) for i in 1:np]

    xf = hcat(sol...)

    return xf, xf
            
end
