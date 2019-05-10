using DifferentialEquations 
using Random

export TimeSeries

mutable struct TimeSeries

   time   :: Vector{Float64}
   values :: Vector{Vector{Float64}}

end

export Catalog

mutable struct Catalog

   analogs    :: Vector{Float64}
   successors :: Vector{Float64}
   sources    :: Vector{Float64}

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

