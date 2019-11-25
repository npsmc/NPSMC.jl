using LinearAlgebra

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

     h        :: Function
     jac_h    :: Function 
     mx       :: Function
     jac_mx   :: Function 

     dt_int   :: Float64
     dt_model :: Int64
     dx       :: Int64
     x0       :: Vector{Float64}
     dy       :: Int64
     var_obs  :: Vector{Int64}
     sig2_Q   :: Float64 
     sig2_R   :: Float64
     Q        :: Symmetric
     R        :: Symmetric


     function SSM( h  :: Function, jac_h  :: Function,
                   mx :: Function, jac_mx :: Function, 
                   dt_int :: Float64, dt_model :: Int64, 
                   x0 :: Vector{Float64}, var_obs :: Vector{Int64}, 
                   sig2_Q :: Float64, sig2_R :: Float64)

        dx       = length(x0)
        dy       = length(var_obs)
        Q        = Symmetric(Matrix(I,dx,dx) .* sig2_Q)
        R        = Symmetric(Matrix(I,dx,dx) .* sig2_R)

        new( h, jac_h, mx, jac_mx, dt_int, dt_model, dx,
             x0, dy, var_obs, sig2_Q, sig2_R, Q, R)

     end

end
