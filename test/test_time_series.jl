@testset " TimeSeries " begin

    using Random
    
    nt, nv = 10, 3
    xt = TimeSeries{Float64}(nt, nv)
    
    @test length(xt.t) == nt
    @test typeof(xt.t) == Array{Float64,1}
    
    t  = collect(0:10.0)
    u  = [rand(nv) for i in eachindex(t)]
    yo = TimeSeries(t, u)
    
    @test typeof(yo.t) == Array{Float64,1}
    @test typeof(yo.u) == Array{Array{Float64,1},1}

end
