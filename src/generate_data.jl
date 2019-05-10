using DifferentialEquations 
using Random

mutable struct TimeSeries

   values :: Vector{Float64}
   time   :: Vector{Float64}

end

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

