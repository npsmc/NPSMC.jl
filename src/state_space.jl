abstract type AbstractForecasting end

export StateSpaceModel

"""

Space-State model is defined through the following equations

```math
\\left\\{
\\begin{array}{l}
X_t = m(X_{t-1}) + \\eta_t, \\\\
Y_t = H(X_t) + \\varepsilon_t,
\\end{array}
\\right.
```

- X : hidden variables
- Y : observed variables

- `dt_integration`is the numerical time step used to solve the ODE.
- `dt_states` is the number of `dt_integration` between ``X_{t-1}`` and ``X_t``.
- `dt_obs` is the number of `dt_integration` between ``Y_{t-1}`` and ``Y_t``.

"""
struct StateSpaceModel <: AbstractForecasting

    model          :: Function
    dt_integration :: Float64
    dt_states      :: Int64
    dt_obs         :: Int64
    params         :: Vector{Float64}
    var_obs        :: Vector{Int64}
    nb_loop_train  :: Int64
    nb_loop_test   :: Int64
    sigma2_catalog :: Float64
    sigma2_obs     :: Float64
    
end

