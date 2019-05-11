using DifferentialEquations 
using Random

export TimeSeries

mutable struct TimeSeries

   time   :: Vector{Float64}
   values :: Array{Union{Float64,Missing},2}

end

export StateSpace

struct StateSpace

    dt_integration :: Float64
    dt_states      :: Int
    dt_obs         :: Int
    params         :: Vector{Float64}
    var_obs        :: Vector{Int64}
    nb_loop_train  :: Int
    nb_loop_test   :: Int
    sigma2_catalog :: Float64
    sigma2_obs     :: Float64
    
end

export Catalog

mutable struct Catalog

   analogs    :: Array{Float64,2}
   successors :: Array{Float64,2}
   sources    :: StateSpace

end
