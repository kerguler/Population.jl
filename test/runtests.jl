using sPop2
using Plots
using Test

@testset "sPop2.jl" begin
    pop = population(acc_data_det(), acc_erlang(), deterministic_update())
    add_pop(pop, 10.0, 0.0)
    step_pop(pop, 20.0, 5.0, 0.0)
end
