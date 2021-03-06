"""
    NPSMC

Non Parametric Sequential Monte-Carlo 
"""
module NPSMC

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

end
