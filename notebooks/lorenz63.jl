# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.3
#   kernelspec:
#     display_name: Julia 1.1.1
#     language: julia
#     name: julia-1.1
# ---

# + {"nbpresent": {"id": "76428090-b279-4d85-b5bc-e0fdefafc294"}, "cell_type": "markdown"}
# # PROBLEM STATEMENT
#
# Data assimilation are numerical methods used in geosciences to mix the information of observations (noted as $y$) and a dynamical model (noted as $f$) in order to estimate the true/hidden state of the system (noted as $x$) at every time step $k$. Usually, they are related following a nonlinear state-space model:
# <img src=https://tandeo.files.wordpress.com/2019/02/formule_nnss_model.png width="200">
# with $\eta$ and $\epsilon$ some independant white Gaussian noises respectively respresenting the model forecast error and the error of observation.
#
# In classical data assimilation, we require multiple runs of an explicit dynamical model $f$ with possible severe limitations including the computational cost, the lack of consistency of the model with respect to the observed data as well as modeling uncertainties. Here, an alternative strategy is explored by developing a fully data-driven assimilation. No explicit knowledge of the dynamical model is required. Only a representative catalog of trajectories of the system is assumed to be available. Based on this catalog, the Analog Data Assimilation (AnDA) is introduced by combining machine learning with the analog method (or nearest neighbor search) and stochastic assimilation techniques including Ensemble Kalman Filter and Smoother (EnKF, EnKS) and Particle Filter (PF). We test the accuracy of the technic on different chaotic dynamical models, the Lorenz-63 and Lorenz-96 systems.
#
# This Julia program is dervied from the Python library is attached to the following publication:
# Lguensat, R., Tandeo, P., Ailliot, P., Pulido, M., & Fablet, R. (2017). The Analog Data Assimilation. *Monthly Weather Review*, 145(10), 4093-4107.
# If you use this library, please do not forget to cite this work.

# + {"nbpresent": {"id": "af657441-0912-4749-b537-6e1734f875bb"}, "cell_type": "markdown"}
# # USING PACKAGES
#
# Here, we import the different Julia packages. In order to use the analog methog (or nearest neighboor search), we need to install the ["NPSMC" library](https://github.com/npsmc/NPSMC.jl).

# + {"nbpresent": {"id": "f975dd20-65cf-43f8-8a6e-96f2acbad4e4"}}
using Plots, DifferentialEquations
# -

include("../src/models.jl")
include("../src/time_series.jl")
include("../src/state_space.jl")
include("../src/catalog.jl")
include("../src/plot.jl")
include("../src/generate_data.jl")
include("../src/utils.jl")
include("../src/model_forecasting.jl")
include("../src/analog_forecasting.jl")
include("../src/data_assimilation.jl")
include("../src/ensemble_kalman_filters.jl")
include("../src/ensemble_kalman_smoothers.jl")
include("../src/particle_filters.jl")

# + {"nbpresent": {"id": "702967c4-5161-4544-a9f1-88cd5d0155da"}, "cell_type": "markdown"}
# # TEST ON LORENZ-63
#
# To begin, as dynamical model $f$, we use the Lorenz-63 chaotic system. First, we generate simulated trajectories from this dynamical model and store them into the catalog. Then, we use this catalog to emulate the dynamical model and we apply the analog data assimilation. Finally, we compare the results of this data-driven approach to the classical data assimilation (using the true Lorenz-63 equations as dynamical model).

# + {"nbpresent": {"id": "81f56606-9081-47fd-8968-13d85c93063c"}}
### GENERATE SIMULATED DATA (LORENZ-63 MODEL)
σ = 10.0
ρ = 28.0
β = 8.0/3

dt_integration = 0.01
dt_states      = 1 
dt_obs         = 8 
parameters     = [σ, ρ, β]
var_obs        = [1]
nb_loop_train  = 100 
nb_loop_test   = 10
sigma2_catalog = 0.0
sigma2_obs     = 2.0

ssm = StateSpaceModel( lorenz63,
                       dt_integration, dt_states, dt_obs, 
                       parameters, var_obs,
                       nb_loop_train, nb_loop_test,
                       sigma2_catalog, sigma2_obs )

# compute u0 to be in the attractor space
u0    = [8.0;0.0;30.0]
tspan = (0.0,5.0)
prob  = ODEProblem(ssm.model, u0, tspan, parameters)
u0    = last(solve(prob, reltol=1e-6, save_everystep=false))

xt, yo, catalog = generate_data( ssm, u0 );

# + {"nbpresent": {"id": "241f9ce2-fe11-4533-be8f-991a700f3920"}}
plot( xt.t, vcat(xt.u'...)[:,1])
scatter!( yo.t, vcat(yo.u'...)[:,1]; markersize=2)

# +
### ANALOG DATA ASSIMILATION (dynamical model given by the catalog)

regression = :local_linear
sampling   = :gaussian
f  = AnalogForecasting( 50, xt, catalog; 
                        regression = regression,
                        sampling   = sampling )
data_assimilation = DataAssimilation( f, xt, ssm.sigma2_obs )
@time x̂_analog  = data_assimilation(yo, EnKS(100))
RMSE(xt, x̂_analog)

# + {"nbpresent": {"id": "3ed39876-5608-4f08-ba3c-a08d1c1d2c84"}}
### CLASSICAL DATA ASSIMILATION (dynamical model given by the equations)
    
data_assimilation = DataAssimilation( ssm, xt )

@time x̂_classical = data_assimilation(yo, EnKS(100))

rmse = RMSE(xt, x̂_classical)


# + {"nbpresent": {"id": "7a6c203f-bcbb-4c52-8b85-7e6be3945044"}}
### COMPARISON BETWEEN CLASSICAL AND ANALOG DATA ASSIMILATION

plot( xt.t, vcat(xt.u'...)[:,1], label="true state")
plot!( xt.t, vcat(x̂_classical.u'...)[:,1], label="classical")
plot!( xt.t, vcat(x̂_analog.u'...)[:,1], label="analog")
scatter!( yo.t, vcat(yo.u'...)[:,1]; markersize=2, label="observations")

# + {"nbpresent": {"id": "971ff88b-e8dc-43dc-897e-71a7b6b659c0"}, "cell_type": "markdown"}
# The results show that performances of the data-driven analog data assimilation are closed to those of the model-driven data assimilation. The error can be reduced by augmenting the size of the catalog "nb_loop_train".

# + {"nbpresent": {"id": "8f5b99c6-6771-4a2f-8ff5-f6693d6b9916"}, "cell_type": "markdown"}
# # Remark
#
# Note that for all the previous experiments, we use the robust Ensemble Kalman Smoother (EnKS) with the increment or local linear regressions and the Gaussian sampling. If you want to have realistic state estimations, we preconize the use of the Particle Filter (DA.method = 'PF') with the locally constant regression (AF.regression = 'locally_constant') and the multinomial sampler (AF.sampling = 'multinomial') with a large number of particles (DA.N). For more details about the different options, see the attached publication: Lguensat, R., Tandeo, P., Ailliot, P., Pulido, M., & Fablet, R. (2017). The Analog Data Assimilation. *Monthly Weather Review*, 145(10), 4093-4107.
