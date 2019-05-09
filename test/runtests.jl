using Test
using NPSMC

@testset "Lorenz 63" begin

@test solve_lorenz63()

end

