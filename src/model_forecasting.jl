""" 
    Apply the dynamical models to generate numerical forecasts. 
"""
function model_forecasting(x, GD)

    # initializations
    N, n    = size(x)
    xf      = zeros(Float64, (N,n))

    tspan = (0.0,5.0)
    p = [10.0,28.0,8/3]
    prob = ODEProblem(lorenz63, x0, tspan, p)

    for i_N in 1:N
        x0 = x[i_N,:]
        S   = odeint(lorenz63, x[i_N,:],
                   np.arange(0,GD.dt_integration+0.000001,GD.dt_integration),
            args=(GD.parameters.sigma,GD.parameters.rho,GD.parameters.beta))
        xf[i_N,:] = S[-1,:]
    end

    xf, xf
            
end
