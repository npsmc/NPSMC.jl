import DifferentialEquations: ODEProblem, EnsembleProblem, solve, SRIW1

""" 

    Apply the dynamical models to generate numerical forecasts. 

"""
function (ssm::StateSpaceModel)(x::Array{Float64,2})

    nv, np = size(x)
    p = ssm.params
    tspan = (0.0, ssm.dt_integration)
    u0 = zeros(Float64, nv)

    function σ( du, u, p, t)

        for i in eachindex(du)
            du[i] = ssm.sigma2_obs
        end

    end

    prob = SDEProblem(ssm.model, σ, u0, tspan, p)

    function prob_func(prob, i, repeat)
        prob.u0 .= x[:, i]
        prob
    end

    monte_prob = EnsembleProblem(prob, prob_func = prob_func)

    sim = solve(monte_prob, SRIW1(); trajectories = np, save_everystep = false)

    sol = [last(sim[i].u) for i = 1:np]

    xf = hcat(sol...)

    return xf, xf

end
