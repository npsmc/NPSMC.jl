include("models.jl")
include("time_series.jl")
include("state_space.jl")
include("catalog.jl")
include("plot.jl")
include("generate_data.jl")
include("utils.jl")
include("model_forecasting.jl")
include("regression.jl")
include("analog_forecasting.jl")
include("data_assimilation.jl")

rng = MersenneTwister(123)
F = 8
J = 40::Int64
parameters = [F, J]
dt_integration = 0.05 # integration time
dt_states = 1 # number of integration times between consecutive states (for xt and catalog)
dt_obs = 4 # number of integration times between consecutive observations (for yo)
var_obs = randperm(rng, J)[1:20] # indices of the observed variables
nb_loop_train = 20 # size of the catalog
nb_loop_test = 5 # size of the true state and noisy observations
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
# -

# 5 time steps (to be in the attractor space)

# +
u0 = F .* ones(Float64, J)
u0[J÷2] = u0[J÷2] + 0.01

tspan = (0.0, 5.0)
p = [F, J]
prob = ODEProblem(lorenz96, u0, tspan, p)
sol = solve(prob, reltol = 1e-6, saveat = dt_integration)
x1 = [x[1] for x in sol.u]
x20 = [x[20] for x in sol.u]
x40 = [x[40] for x in sol.u]

# run the data generation
xt, yo, catalog = generate_data(ssm, u0);

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

neighborhood = local_analog_matrix
regression = :local_linear
sampling = :gaussian
f = AnalogForecasting(100, xt, catalog, neighborhood, regression, sampling)
DA_local = DataAssimilation(f, xt, ssm.sigma2_obs)
@time x̂_analog_local = forecast(DA_local, yo, EnKS(500))

f = AnalogForecasting(100, xt, catalog, regression = regression, sampling = sampling)
DA_global = DataAssimilation(f, xt, ssm.sigma2_obs)
@time x̂_analog_global = forecast(DA_global, yo, EnKS(500))
println(RMSE(xt, x̂_analog_global))

# # error
println("RMSE(global analog DA) = $(RMSE(xt,x̂_analog_global))")
println("RMSE(local analog DA)  = $(RMSE(xt,x̂_analog_local))")
