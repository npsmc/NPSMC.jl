# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Julia 1.3.1
#     language: julia
#     name: julia-1.3
# ---

# + [markdown] {"nbpresent": {"id": "76428090-b279-4d85-b5bc-e0fdefafc294"}}
# # Problem
#
# Data assimilation are numerical methods used in geosciences to mix the information of observations (noted as $y$) and a dynamical model (noted as $f$) in order to estimate the true/hidden state of the system (noted as $x$) at every time step $k$. Usually, they are related following a nonlinear state-space model:
# <img src=https://tandeo.files.wordpress.com/2019/02/formule_nnss_model.png width="200">
# with $\eta$ and $\epsilon$ some independant white Gaussian noises respectively respresenting the model forecast error and the error of observation.
#
# In classical data assimilation, we require multiple runs of an explicit dynamical model $f$ with possible severe limitations including the computational cost, the lack of consistency of the model with respect to the observed data as well as modeling uncertainties. Here, an alternative strategy is explored by developing a fully data-driven assimilation. No explicit knowledge of the dynamical model is required. Only a representative catalog of trajectories of the system is assumed to be available. Based on this catalog, the Analog Data Assimilation (AnDA) is introduced by combining machine learning with the analog method (or nearest neighbor search) and stochastic assimilation techniques including Ensemble Kalman Filter and Smoother (EnKF, EnKS) and Particle Filter (PF). We test the accuracy of the technic on chaotic dynamical models, the Lorenz-96 system.
#
# This Julia program is dervied from the Python library is attached to the following publication:
# Lguensat, R., Tandeo, P., Ailliot, P., Pulido, M., & Fablet, R. (2017). The Analog Data Assimilation. *Monthly Weather Review*, 145(10), 4093-4107.
# If you use this library, please do not forget to cite this work.

# + [markdown] {"nbpresent": {"id": "af657441-0912-4749-b537-6e1734f875bb"}}
# # NPSMC.jl package
#
# Here, we import the different Julia packages. In order to use the analog methog (or nearest neighboor search), we need to install the ["NPSMC" library](https://github.com/npsmc/NPSMC.jl).

# + {"nbpresent": {"id": "f975dd20-65cf-43f8-8a6e-96f2acbad4e4"}}
using Plots, NPSMC, DifferentialEquations, Random, LinearAlgebra

# + [markdown] {"nbpresent": {"id": "c4e459e9-33bc-43f1-91e8-5a5d05746979"}}
# # TEST ON LORENZ-96
#
# We also test the analog data assimilation procedure on the 40-dimensional Lorenz-96 dynamical model. As in the previous experiment, we generate state and observation data as well as simulated trajectories of the Lorenz-96 model in order to emulate the dynamical model. Here, we compare two analog data assimilation strategies: the global and local analog forecasting, respectively defined in finding similar situations on the whole 40 variables or on 5 variables recursively.

# +
rng = MersenneTwister(123)
F = 8
J = 40 :: Int64
parameters = [F, J]
dt_integration = 0.05 # integration time
dt_states = 1 # number of integration times between consecutive states (for xt and catalog)
dt_obs = 4 # number of integration times between consecutive observations (for yo)
var_obs = randperm(rng, J)[1:20] # indices of the observed variables
nb_loop_train = 100 # size of the catalog
nb_loop_test = 10 # size of the true state and noisy observations
sigma2_catalog = 0.   # variance of the model error to generate the catalog   
sigma2_obs = 2. # variance of the observation error to generate observations

ssm = StateSpaceModel( lorenz96, 
                       dt_integration, dt_states, dt_obs, 
                       parameters, var_obs,
                       nb_loop_train, nb_loop_test,
                       sigma2_catalog, sigma2_obs )
# -

# 5 time steps (to be in the attractor space)

# +
u0 = F .* ones(Float64, J)
u0[J÷2] = u0[J÷2] + 0.01

tspan = (0.0,5.0)
p = [F, J]
prob  = ODEProblem(lorenz96, u0, tspan, p)
sol = solve(prob, reltol=1e-6, saveat= dt_integration)
x1  = getindex.(sol.u, 1)
x20 = getindex.(sol.u, 20)
x40 = getindex.(sol.u, 40)
u0 = last(sol.u);
plot(sol.t, [x1 x20 x40], labels=[:x1 :x20 :x40])
# -

# run the data generation
xt, yo, catalog = generate_data(ssm, u0);

# ### PLOT STATE, OBSERVATIONS AND CATALOG

plot(xt.t,  xt[1], line=(:solid,:red), label="x1")
scatter!(yo.t, yo[1], markersize=2)
plot!(xt.t, xt[20], line=(:solid,:blue), label="x20")
scatter!(yo.t, yo[20], markersize=2)
plot!(xt.t, xt[40], line=(:solid,:green), label="x40")
scatter!(yo.t, yo[40], markersize=2)
xlabel!("Lorenz-96 times")
title!("Lorenz-96 true (continuous lines) and observed trajectories (dots)")

# ### MODEL DATA ASSIMILATION (with the global analogs)

DA = DataAssimilation( ssm, xt )
@time x̂_classical_global  = forecast(DA, yo, EnKS(500));

RMSE(xt, x̂_classical_global)

# + {"nbpresent": {"id": "604a659e-82bf-4618-95bf-77ef755b9088"}}
local_analog_matrix =  BitArray{2}(diagm( -2  => trues(xt.nv-2),
             -1  => trues(xt.nv-1),
              0  => trues(xt.nv),
              1  => trues(xt.nv-1),
              2  => trues(xt.nv-2),             
             J-2 => trues(xt.nv-(J-2)),
             J-1 => trues(xt.nv-(J-1)))
    + transpose(diagm( J-2 => trues(xt.nv-(J-2)),
             J-1 => trues(xt.nv-(J-1))))
    );
# -

heatmap(local_analog_matrix) 

# + [markdown] {"nbpresent": {"id": "150c861c-fecc-4dfc-8bb7-c54189d675cb"}}
# To define the local or global analog forecasting, we generate different matrices that will be use as the "AF.neighborhood" argument. For each variable of the system, we use 0 or 1 to indicate the absence or presence of other variables in the analog forecasting procedure. For instance, in the local analog matrix defined above, to predict the variable $x_2$ at time t+dt, we will use the local variables $x_1$, $x_2$, $x_3$, $x_4$ and $x_{40}$ at time t.
# -

# ### ANALOG DATA ASSIMILATION (with the global analogs)

regression = :local_linear
sampling = :gaussian
f  = AnalogForecasting( 100, xt, catalog, regression=regression, sampling=sampling)
DA_global = DataAssimilation( f, xt, ssm.sigma2_obs )
@time x̂_analog_global  = forecast(DA_global, yo, EnKS(500))
println(RMSE(xt, x̂_analog_global))

DA = DataAssimilation( f, xt, ssm.sigma2_obs )
@time x̂_analog_global  = forecast(DA, yo, EnKS(500))
RMSE(xt, x̂_analog_global)

# + [markdown] {"nbpresent": {"id": "02cf2959-e712-4af8-8bb6-f914608e15ac"}}
# ### ANALOG DATA ASSIMILATION (with the local analogs)

# + {"nbpresent": {"id": "02cf2959-e712-4af8-8bb6-f914608e15ac"}}
neighborhood = local_analog_matrix
regression = :local_linear
sampling   = :gaussian
f  = AnalogForecasting( 100, xt, catalog, neighborhood, regression, sampling)
DA = DataAssimilation( f, xt, ssm.sigma2_obs )
@time x̂_analog_local  = forecast(DA, yo, EnKS(500))
RMSE(xt, x̂_analog_local)

# + [markdown] {"nbpresent": {"id": "35f54171-6e87-4b0f-9cb2-821d9c0d8b96"}}
# ### COMPARISON BETWEEN GLOBAL AND LOCAL ANALOG DATA ASSIMILATION

# + {"nbpresent": {"id": "35f54171-6e87-4b0f-9cb2-821d9c0d8b96"}}
import PyPlot
fig = PyPlot.figure(figsize=(10,10))
# plot
PyPlot.subplot(221)
PyPlot.pcolormesh(hcat(xt.u...))
PyPlot.ylabel("Lorenz-96 times")
PyPlot.title("True trajectories")
PyPlot.subplot(222)
PyPlot.pcolormesh(isnan.(hcat(yo.u...)))
PyPlot.ylabel("Lorenz-96 times")
PyPlot.title("Observed trajectories")
PyPlot.subplot(223)
PyPlot.pcolormesh(hcat(x̂_analog_global.u...))
PyPlot.ylabel("Lorenz-96 times")
PyPlot.title("Global analog data assimilation")
PyPlot.subplot(224)
PyPlot.pcolormesh(hcat(x̂_analog_local.u...))
PyPlot.ylabel("Lorenz-96 times")
PyPlot.title("Local analog data assimilation")

# + [markdown] {"nbpresent": {"id": "35f54171-6e87-4b0f-9cb2-821d9c0d8b96"}}
# # error

# + {"nbpresent": {"id": "35f54171-6e87-4b0f-9cb2-821d9c0d8b96"}}
println("RMSE(global analog DA) = $(RMSE(xt,x̂_analog_global))")
println("RMSE(local analog DA)  = $(RMSE(xt,x̂_analog_local))")

# + [markdown] {"nbpresent": {"id": "3c8d57d0-c7b5-4ec6-9ecb-d91aaffbf836"}}
# The results show that the global analog strategy do not reach satisfying results. Indeed, it is difficult to find relevant nearest neighboors on 40-dimensional vectors. The only way to improve the results in such a global strategy is to deeply increase the size of the catalog. At the contrary, in the local analog data assimilation, we are able to track correctly the true trajectories, even with a short catalog.

# + [markdown] {"nbpresent": {"id": "8f5b99c6-6771-4a2f-8ff5-f6693d6b9916"}}
# # Remark
#
# Note that for all the previous experiments, we use the robust Ensemble Kalman Smoother (EnKS) with the increment or local linear regressions and the Gaussian sampling. If you want to have realistic state estimations, we preconize the use of the Particle Filter (DA.method = 'PF') with the locally constant regression (AF.regression = 'locally_constant') and the multinomial sampler (AF.sampling = 'multinomial') with a large number of particles (DA.N). For more details about the different options, see the attached publication: Lguensat, R., Tandeo, P., Ailliot, P., Pulido, M., & Fablet, R. (2017). The Analog Data Assimilation. *Monthly Weather Review*, 145(10), 4093-4107.
