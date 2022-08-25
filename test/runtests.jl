using sPop2
using Plots
using Test

@testset "sPop2.jl" begin
    pop = Population(PopDataSto(), AccErlang())
    AddPop(pop, 10.0, 0, 0.0)
    StepPop(pop, 20.0, 5.0, 0.0)
end
