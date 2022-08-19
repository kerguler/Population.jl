using sPop2
using Plots
using Test

@testset "sPop2.jl" begin
    pop = newPop(false, "ACC_ERLANG")
    addPop(pop, 0, 0.0, 10.0)
    stepPop(pop, 20.0, 5.0, 0.0)
end
