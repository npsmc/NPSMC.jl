export TimeSeries

mutable struct TimeSeries{T}

   nt     :: Integer
   nv     :: Integer
   time   :: Vector{T}
   values :: Array{T, 2}

   function TimeSeries{T}( nt :: Integer, nv :: Integer) where T
 
       time   = zeros(T, nt)
       values = zeros(T, (nv, nt))

       new( nt, nv, time, values)

   end

   function TimeSeries( time   :: Array{T, 1}, 
                        values :: Array{T, 2}) where T
 
       nt = length(time)
       nv = first(size(values))

       new{T}( nt, nv, time, values)

   end

end

