# # Lorenz 96

using Plots, DifferentialEquations, Random, LinearAlgebra
using SparseArrays
using NPSMC

# We test the analog data assimilation procedure on the 40-dimensional
# Lorenz-96 dynamical model. As in the [Lorenz 63](@ref) experiment, 
# we generate
# state and observation data as well as simulated trajectories of the
# Lorenz-96 model in order to emulate the dynamical model. Here, we
# compare two analog data assimilation strategies: the global and
# local analog forecasting, respectively defined in finding similar
# situations on the whole 40 variables or on 5 variables recursively.

rng = MersenneTwister(123)
F = 8
J = 40::Int64
parameters = [F, J]
dt_integration = 0.05 # integration time
dt_states = 1 # number of integration times between consecutive states (for xt and catalog)
dt_obs = 4 # number of integration times between consecutive observations (for yo)
var_obs = randperm(rng, J)[1:20] # indices of the observed variables
nb_loop_train = 100 # size of the catalog
nb_loop_test = 10 # size of the true state and noisy observations
sigma2_catalog = 0.0   # variance of the model error to generate the catalog   
sigma2_obs = 2.0 # variance of the observation error to generate observations

ssm = StateSpaceModel(
    lorenz96,
    dt_integration,
    dt_states,
    dt_obs,
    parameters,
    var_obs,
    nb_loop_train,
    nb_loop_test,
    sigma2_catalog,
    sigma2_obs,
)

# 5 time steps (to be in the attractor space)

u0 = F .* ones(Float64, J)
u0[J÷2] = u0[J÷2] + 0.01

tspan = (0.0, 5.0)
p = [F, J]
prob = ODEProblem(lorenz96, u0, tspan, p)
sol = solve(prob, reltol = 1e-6, saveat = dt_integration)
x1 = [x[1] for x in sol.u]
x20 = [x[20] for x in sol.u]
x40 = [x[40] for x in sol.u]
plot(sol.t, x1)
plot!(sol.t, x20)
plot!(sol.t, x40)

# run the data generation

xt, yo, catalog = generate_data(ssm, u0);

# ## Plot state, observations and catalog

plot(xt.t, xt[1], line = (:solid, :red), label = "x1")
scatter!(yo.t, yo[1], markersize = 2)
plot!(xt.t, xt[20], line = (:solid, :blue), label = "x20")
scatter!(yo.t, yo[20], markersize = 2)
plot!(xt.t, xt[40], line = (:solid, :green), label = "x40")
scatter!(yo.t, yo[40], markersize = 2)
xlabel!("Lorenz-96 times")
title!("Lorenz-96 true (continuous lines) and observed trajectories (dots)")

# ## Model data assimilation (with the global analogs)

DA = DataAssimilation(ssm, xt)
@time x̂_classical_global = forecast(DA, yo, EnKS(500), progress = false);

# - RMSE

RMSE(xt, x̂_classical_global)

# - Set the local analog matrix

local_analog_matrix = BitArray{2}(diagm(
    -2 => trues(xt.nv - 2),
    -1 => trues(xt.nv - 1),
    0 => trues(xt.nv),
    1 => trues(xt.nv - 1),
    2 => trues(xt.nv - 2),
    J - 2 => trues(xt.nv - (J - 2)),
    J - 1 => trues(xt.nv - (J - 1)),
) + transpose(diagm(
    J - 2 => trues(xt.nv - (J - 2)),
    J - 1 => trues(xt.nv - (J - 1)),
)));

spy(sparse(local_analog_matrix))

# To define the local or global analog forecasting, we generate
# different matrices that will be use as the "AF.neighborhood" argument.
# For each variable of the system, we use 0 or 1 to indicate the
# absence or presence of other variables in the analog forecasting
# procedure. For instance, in the local analog matrix defined above,
# to predict the variable ``x_2`` at time t+dt, we will use the local
# variables ``x_1``, ``x_2``, ``x_3``, ``x_4`` and ``x_{40}`` at time t.  # -

# ## Analog data assimilation (with the global analogs)

f = AnalogForecasting(
    100,
    xt,
    catalog,
    regression = :locally_constant,
    sampling = :gaussian,
)

DA = DataAssimilation(f, xt, ssm.sigma2_obs)

@time x̂_analog_global = forecast(DA, yo, EnKS(500), progress = false)
RMSE(xt, x̂_analog_global)

# ## Analog data assimilation (with the local analogs)

neighborhood = local_analog_matrix
regression = :local_linear
sampling = :gaussian

f = AnalogForecasting(100, xt, catalog, neighborhood, regression, sampling)

DA = DataAssimilation(f, xt, ssm.sigma2_obs)
@time x̂_analog_local = forecast(DA, yo, EnKS(500), progress = false)
RMSE(xt, x̂_analog_local)

# ## Comparison between global and local analog data assimilation

import PyPlot
fig = PyPlot.figure(figsize = (10, 10))

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

# - Error

println("RMSE(global analog DA) = $(RMSE(xt,x̂_analog_global))")
println("RMSE(local analog DA)  = $(RMSE(xt,x̂_analog_local))")

# The results show that the global analog strategy do not reach
# satisfying results. Indeed, it is difficult to find relevant nearest
# neighboors on 40-dimensional vectors. The only way to improve the
# results in such a global strategy is to deeply increase the size
# of the catalog. At the contrary, in the local analog data assimilation,
# we are able to track correctly the true trajectories, even with a
# short catalog.
