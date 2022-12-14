using Population
using Test

@testset "Population.jl" begin
    pop = sPop2(PopDataSto())
    AddProcess(pop, AccErlang())
    AddPop(pop, 10)
    @test GetPop(pop) == 10
    @test length(GetPoptable(pop)) == 1
    @test length(GetPoptable(pop)[1]) == 1
    @test GetPoptable(pop)[1][0.0] == 10
    pr = (devmn=20.0, devsd=5.0)
    StepPop(pop, pr)
end