using LinearAlgebra

abstract type AbstractForecasting end

export StateSpaceModel

"""

Space-State model is defined through the following equations

Let ${x_0, x_1, \dots, x_K }$ denote the discrete-time state process,
which is typically unknown to the observer. Let ${y_1,y_2,... ,y_K}$ 
represent the observation process, which describes the measurements of the
system that are available to the observer. 

These state and observation processes are related following a nonlinear 
state-space model: The first equation describes how the system state $x_k$ 
evolves through a nonlinear dynamical model $f_{k−1,k}$ between successive 
time steps $t_{k−1}$ and $t_k$.
$η = {η1,η2,... ,ηK}$ is the model noise process, which accounts for
the imperfections of the model; it is assumed to be independent and
identically distributed. At each time step, $η_k$ is assumed
to be Gaussian with zero mean and covariance matrix Q. The second equation
models how an observation $y_k$ at time $t_k$ is obtained from the state
$x_k$ through a linear operator $H$. $ε = {ε1, ε2, . . . , εk}$ is an
independent and identically distributed process representing the observation errors; εk is assumed
Gaussian with zero mean and covariance matrix R. Finally, the
processes η and ε are assumed to be jointly independent and independent
of the initial (background) state x0, which is assumed to be Gaussian
with mean xb and covariance B.

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

    model::Function
    dt_integration::Float64
    dt_states::Int64
    dt_obs::Int64
    params::Vector{Float64}
    var_obs::Vector{Int64}
    nb_loop_train::Int64
    nb_loop_test::Int64
    sigma2_catalog::Float64
    sigma2_obs::Float64

end



"""
    Generate simulated data from Space State Model
     
- var_obs            : indices of the observed variables
- dy                 : dimension of the observations
- Q                  : model covariance
- R                  : observation covariance
- dx                 : dimension of the state
- dt_int             : fixed integration time
- dt_model           : chosen number of model time step 
- var_obs            : indices of the observed variables
- dy                 : dimension of the observations
- H                  : first and third variables are observed
- h                  : observation model
- jacH               : Jacobian of the observation model(for EKS_EM only)
- Q                  : model covariance
- R                  : observation covariance


- Example [Sinus data](@ref)

"""
struct SSM

    h::Function
    jac_h::Function
    mx::Function
    jac_mx::Function

    dt_int::Float64
    dt_model::Int64
    dx::Int64
    x0::Vector{Float64}
    dy::Int64
    var_obs::Vector{Int64}
    sig2_Q::Float64
    sig2_R::Float64
    Q::Symmetric
    R::Symmetric


    function SSM(
        h::Function,
        jac_h::Function,
        mx::Function,
        jac_mx::Function,
        dt_int::Float64,
        dt_model::Int64,
        x0::Vector{Float64},
        var_obs::Vector{Int64},
        sig2_Q::Float64,
        sig2_R::Float64,
    )

        dx = length(x0)
        dy = length(var_obs)
        Q = Symmetric(Matrix(I, dx, dx) .* sig2_Q)
        R = Symmetric(Matrix(I, dx, dx) .* sig2_R)

        new(
            h,
            jac_h,
            mx,
            jac_mx,
            dt_int,
            dt_model,
            dx,
            x0,
            dy,
            var_obs,
            sig2_Q,
            sig2_R,
            Q,
            R,
        )

    end

end
