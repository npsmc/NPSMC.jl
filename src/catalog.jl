export Catalog

"""
    Catalog( data, ssm)

Data type to store analogs and succesors observations
from the Space State model

- Example [Example Catalog](@ref)
"""
struct Catalog

   nt         :: Int64
   nv         :: Int64
   data       :: Array{Float64,2}
   analogs    :: Array{Float64,2}
   successors :: Array{Float64,2}
   sources    :: StateSpaceModel

   function Catalog( data :: Array{Float64,2}, ssm :: StateSpaceModel)

       nv, nt     = size(data)
       analogs    = data[ :, 1:end-ssm.dt_states]
       successors = data[ :, ssm.dt_states+1:end]

       new( nt, nv, data, analogs, successors, ssm)

   end

end
