using sPop2
using Plots
using Test

@testset "sPop2.jl" begin
    pop = Population(PopDataSto())
    AddProcess(pop, AccErlang())
    AddPop(pop, 10)
    pr = (devmn=20.0, devsd=5.0)
    StepPop(pop, pr)
end
