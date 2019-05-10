using DifferentialEquations 
using Random

export TimeSeries

mutable struct TimeSeries

   time   :: Vector{Float64}
   values :: Array{Float64,2}

end

export Catalog

mutable struct Catalog

   analogs    :: Array{Float64,2}
   successors :: Array{Float64,2}
   sources    :: Array{Float64,2}

end

export StateSpace

struct StateSpace

    dt_states      :: Float64
    dt_obs         :: Float64
    params         :: Vector{Float64}
    var_obs        :: Vector{Int64}
    nb_loop_train  :: Int64
    nb_loop_test   :: Int64
    sigma2_catalog :: Float64
    sigma2_obs     :: Float64
    
end

