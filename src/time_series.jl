abstract type AbstractTimeSeries end

export TimeSeries


mutable struct TimeSeries{T} <:  AbstractTimeSeries

   nt  :: Integer
   nv  :: Integer
   t   :: Vector{T}
   u   :: Vector{Array{T, 1}}

   function TimeSeries{T}( nt :: Integer, nv :: Integer) where T
 
       time   = zeros(T, nt)
       values = [zeros(T, nv) for i in 1:nt]

       new( nt, nv, time, values)

   end

   function TimeSeries( time   :: Array{T, 1}, 
                        values :: Array{Array{T, 1}}) where T
 
       nt = length(time)
       nv = size(first(values))[1]

       new{T}( nt, nv, time, values)

   end

end

