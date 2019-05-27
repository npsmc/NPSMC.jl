export Catalog

mutable struct Catalog

   nt         :: Int64
   nv         :: Int64
   data       :: Array{Float64,2}
   analogs    :: Array{Float64,2}
   successors :: Array{Float64,2}
   sources    :: StateSpaceModel

   function Catalog( data :: Array{Float64,2}, ssm :: StateSpaceModel)

       nv, nt     = size(data)
       analogs    = @view data[ :, 1:end-ssm.dt_states]
       successors = @view data[ :, ssm.dt_states+1:end]

       new( nt, nv, data, analogs, successors, ssm)

   end

end
