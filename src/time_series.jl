abstract type AbstractTimeSeries end

export TimeSeries

struct TimeSeries <: AbstractTimeSeries

    nt::Integer
    nv::Integer
    t::Vector{Float64}
    u::Vector{Array{Float64,1}}

    function TimeSeries(nt::Integer, nv::Integer)

        time = zeros(Float64, nt)
        values = [zeros(Float64, nv) for i = 1:nt]

        new(nt, nv, time, values)

    end

    function TimeSeries(time::Array{Float64,1}, values::Array{Array{Float64,1}})

        nt = length(time)
        nv = size(first(values))[1]

        new(nt, nv, time, values)

    end

end

import Base: getindex

function getindex(x::TimeSeries, i::Int)
    getindex.(x.u, i)
end

import Base:-

function (-) (a :: TimeSeries, b :: TimeSeries)

    @assert a.nv == b.nv
    @assert all(a.t .== b.t)

    TimeSeries(a.t, [a[i] - b[i] for i in 1:a.nt])

end
