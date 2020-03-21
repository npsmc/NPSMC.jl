struct Ensemble

    nv :: Int64
    ne :: Int64
    nt :: Int64
    data :: Array{Array{Float64,2},1}
    
    function Ensemble( nv, ne, nt)
        data = [zeros(nv, ne) for i in 1:nt]
        new(nv, ne, nt, data)
    end
    
end

import Statistics: mean

function mean(x :: Ensemble)
    
    return [ vec(mean(x[i], dims=2)) for i in 1:x.nt]
    
end
