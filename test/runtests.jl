using sPop2
using Plots
using Test

@testset "sPop2.jl" begin
    pop = Population(PopDataSto(), AccErlang())
    add_pop(pop, 10.0, 0, 0.0)
    step_pop(pop, 20.0, 5.0, 0.0)
end
