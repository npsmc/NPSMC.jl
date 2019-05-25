export Catalog

mutable struct Catalog{T}

   nt         :: Int64
   nv         :: Int64
   data       :: Array{T,2}
   analogs    :: Array{T,2}
   successors :: Array{T,2}
   sources    :: StateSpaceModel

   function Catalog( data :: Array{T,2}, ssm :: StateSpaceModel) where T

       nv, nt     = size(data)
       analogs    = @view data[ :, 1:end-ssm.dt_states]
       successors = @view data[ :, ssm.dt_states+1:end]

       new{T}( nt, nv, data, analogs, successors, ssm)

   end

end
