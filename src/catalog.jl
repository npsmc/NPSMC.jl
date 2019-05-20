export Catalog

mutable struct Catalog{T}

   data       :: Array{T,2}
   analogs    :: AbstractArray{T,2}
   successors :: AbstractArray{T,2}
   sources    :: StateSpaceModel

   function Catalog( data :: Array{T,2}, ssm :: StateSpaceModel) where T

       analogs    = @view data[1:end-ssm.dt_states,:]
       successors = @view data[ssm.dt_states+1:end,:]

       new{T}( data, analogs, successors, ssm)

   end

end

