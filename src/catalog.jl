export Catalog

mutable struct Catalog{T}

   nt         :: Int64
   nv         :: Int64
   data       :: Vector{Array{T,1}}
   analogs    :: Vector{Array{T,1}}
   successors :: Vector{Array{T,1}}
   sources    :: StateSpaceModel

   function Catalog( data :: Vector{Array{T,1}}, ssm :: StateSpaceModel) where T

       nt, nv     = length(data), length(data[1])
       analogs    = @view data[1:end-ssm.dt_states]
       successors = @view data[ssm.dt_states+1:end]

       new{T}( nt, nv, data, analogs, successors, ssm)

   end

end

