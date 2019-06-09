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

# + {"nbpresent": {"id": "c4e459e9-33bc-43f1-91e8-5a5d05746979"}, "cell_type": "markdown"}
# # TEST ON LORENZ-96
#
# We also test the analog data assimilation procedure on the 40-dimensional Lorenz-96 dynamical model. As in the previous experiment, we generate state and observation data as well as simulated trajectories of the Lorenz-96 model in order to emulate the dynamical model. Here, we compare two analog data assimilation strategies: the global and local analog forecasting, respectively defined in finding similar situations on the whole 40 variables or on 5 variables recursively.

# +
using Random

F = 8
J = 40 :: Int64
parameters = [F, J]
dt_integration = 0.05 # integration time
dt_states = 1 # number of integration times between consecutive states (for xt and catalog)
dt_obs = 4 # number of integration times between consecutive observations (for yo)
var_obs = randperm(MersenneTwister(1234), J)[1:20] # indices of the observed variables
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
# Checks if the inegration function is well implemented.

# +
u0 = F .* ones(Float64, J)
u0[J÷2] = u0[J÷2] + 0.01

tspan = (0.0,5.0)
p = [F, J]
prob  = ODEProblem(lorenz96, u0, tspan, p)
sol = solve(prob, reltol=1e-6, saveat= dt_integration)
x1  = [x[1] for x in sol.u]
x20 = [x[20] for x in sol.u]
x40 = [x[40] for x in sol.u]
plot(sol.t, x1)
plot!(sol.t, x20)
plot!(sol.t, x40)
# -

# run the data generation
xt, yo, catalog = generate_data(ssm, u0);

# +
### PLOT STATE, OBSERVATIONS AND CATALOG

plot(xt.t,  vcat(xt.u'...)[:,1], line=(:solid,:red), label="x1")
scatter!(yo.t, vcat(yo.u'...)[:,1], markersize=2)
plot!(xt.t, vcat(xt.u'...)[:,20],line=(:solid,:blue), label="x20")
scatter!(yo.t, vcat(yo.u'...)[:,20],markersize=2)
plot!(xt.t, vcat(xt.u'...)[:,40],line=(:solid,:green), label="x40")
scatter!(yo.t, vcat(yo.u'...)[:,40],markersize=2)
xlabel!("Lorenz-96 times")
title!("Lorenz-96 true (continuous lines) and observed trajectories (dots)")

# + {"nbpresent": {"id": "604a659e-82bf-4618-95bf-77ef755b9088"}}
using LinearAlgebra

# + {"nbpresent": {"id": "604a659e-82bf-4618-95bf-77ef755b9088"}}
local_analog_matrix = BitArray{2}( diagm( -2  => trues(xt.nv-2),
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

# + {"nbpresent": {"id": "150c861c-fecc-4dfc-8bb7-c54189d675cb"}, "cell_type": "markdown"}
# To define the local or global analog forecasting, we generate different matrices that will be use as the "AF.neighborhood" argument. For each variable of the system, we use 0 or 1 to indicate the absence or presence of other variables in the analog forecasting procedure. For instance, in the local analog matrix defined above, to predict the variable $x_2$ at time t+dt, we will use the local variables $x_1$, $x_2$, $x_3$, $x_4$ and $x_{40}$ at time t.
# -

### ANALOG DATA ASSIMILATION (with the global analogs)
include("../src/analog_forecasting.jl")
include("../src/data_assimilation.jl")
f  = AnalogForecasting( 50, xt, catalog, 
    regression = :local_linear, sampling   = :gaussian)
data_assimilation = DataAssimilation( f, xt, ssm.sigma2_obs )
@time x̂_analog  = data_assimilation(yo, EnKS(100))
RMSE(xt, x̂_analog)

# + {"nbpresent": {"id": "02cf2959-e712-4af8-8bb6-f914608e15ac"}}
### ANALOG DATA ASSIMILATION (with the local analogs)

neighborhood = local_analog_matrix
regression = :local_linear
sampling   = :gaussian
f  = AnalogForecasting( 100, xt, catalog, neighborhood, regression, sampling)
data_assimilation = DataAssimilation( f, xt, ssm.sigma2_obs )
@time x̂  = data_assimilation(yo, EnKS(500))
RMSE(xt, x̂)

# +
import PyPlot

#####################
##  2x2 Plot Grid  ##
#####################
fig = figure("pyplot_subplot_mixed",figsize=(10,10)) # Create a new blank figure
#fig.set_figheight(7) # Doesn't work
#fig.set_figwidth(3) # Doesn't work
subplot(221) # Create the 1st axis of a 2x2 arrax of axes
grid("on") # Create a grid on the axis
PyPlot.title("221") # Give the most recent axis a title
subplot(222,polar="true") # Create a plot and make it a polar plot, 2nd axis of 2x2 axis grid
PyPlot.title("222")
ax = subplot(223,polar="true") # Create a plot and make it a polar plot, 3rd axis of 2x2 axis grid
ax.set_theta_zero_location("N") # Set 0 degrees to the top of the plot
ax.set_theta_direction(-1) # Switch the polar plot to clockwise
PyPlot.title("223")
subplot(224) # Create the 4th axis of a 2x2 arrax of axes
xlabel("This is an X axis")
ylabel("This is a y axis")
PyPlot.title("224")
fig.canvas.draw() # Update the figure
suptitle("2x2 Subplot")

# + {"nbpresent": {"id": "35f54171-6e87-4b0f-9cb2-821d9c0d8b96"}}



# + {"nbpresent": {"id": "35f54171-6e87-4b0f-9cb2-821d9c0d8b96"}}
### COMPARISON BETWEEN GLOBAL AND LOCAL ANALOG DATA ASSIMILATION
fig = figure(figsize=(10,10))
# plot
pcolormesh(hcat(xt.u...))
ylabel('Lorenz-96 times')
PyPlot.title('True trajectories')
subplot(222)
pcolormesh(isnan.(hcat(yo.u..)))
ylabel('Lorenz-96 times')
PyPlot.title('Observed trajectories')
subplot(223)
pcolormesh(x̂_analog_global.u)
ylabel('Lorenz-96 times')
PyPlot.title('Global analog data assimilation')
subplot(224)
pcolormesh(hcat(x̂_analog_local.u...))
ylabel('Lorenz-96 times')
PyPlot.title('Local analog data assimilation')

# + {"nbpresent": {"id": "35f54171-6e87-4b0f-9cb2-821d9c0d8b96"}}
# error
print("RMSE(global analog DA) = $(RMSE(xt,x̂_analog_global))")
print("RMSE(local analog DA)  = $(RMSE(xt,x̂_analog_local))")

# + {"nbpresent": {"id": "3c8d57d0-c7b5-4ec6-9ecb-d91aaffbf836"}, "cell_type": "markdown"}
# The results show that the global analog strategy do not reach satisfying results. Indeed, it is difficult to find relevant nearest neighboors on 40-dimensional vectors. The only way to improve the results in such a global strategy is to deeply increase the size of the catalog. At the contrary, in the local analog data assimilation, we are able to track correctly the true trajectories, even with a short catalog.

# + {"nbpresent": {"id": "8f5b99c6-6771-4a2f-8ff5-f6693d6b9916"}, "cell_type": "markdown"}
# # Remark
#
# Note that for all the previous experiments, we use the robust Ensemble Kalman Smoother (EnKS) with the increment or local linear regressions and the Gaussian sampling. If you want to have realistic state estimations, we preconize the use of the Particle Filter (DA.method = 'PF') with the locally constant regression (AF.regression = 'locally_constant') and the multinomial sampler (AF.sampling = 'multinomial') with a large number of particles (DA.N). For more details about the different options, see the attached publication: Lguensat, R., Tandeo, P., Ailliot, P., Pulido, M., & Fablet, R. (2017). The Analog Data Assimilation. *Monthly Weather Review*, 145(10), 4093-4107.
