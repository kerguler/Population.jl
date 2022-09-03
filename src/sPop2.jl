#=
sPop2: a dynamically-structured matrix Population model
Copyright (C) 2022 Kamil Erguler <k.erguler@cyi.ac.cy>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
=#

module sPop2

export AccHaz, AgeHaz, CusHaz, HazTypes,
       AccFixed, AccPascal, AccErlang,
       AgeFixed, AgeNbinom, AgeGamma, 
       AgeConst, AgeCustom, AgeDummy,
       PopDataDet, PopDataSto, Population, 
       StepPop, AddPop, GetPop, MemberKey,
       set_acc_eps, EmptyPop, GetPoptable,
       AddProcess, AccStepper, AgeStepper, CustomStepper,
       StepperTypes

using Distributions
using Random: rand

const ACCTHR = 1.0

EPS = 14
function set_acc_eps(eps::Int64)
    global EPS
    EPS = eps == 0 ? 14 : eps
    return EPS
end

# --------------------------------------------------------------------------------
# hazard type
# --------------------------------------------------------------------------------

abstract type HazTypes end
abstract type AccHaz <: HazTypes end
abstract type AgeHaz <: HazTypes end
abstract type CusHaz <: HazTypes end

"""
Hazard Calculation for Accumulative Process

The hazard is computed as ``\\frac{F(x,k,θ) - F(x-1,k,θ)}{1 - F(x-1,k,θ)} ``

"""
function acc_hazard_calc(heval::Function, d::Number, q::Number, k::Number, theta::Number, qkey::Tuple)
    h0 = d == zero(d) ? 0.0 : heval(d - one(d), k, theta)
    h1 = heval(d, k, theta)
    h0 == 1.0 ? 1.0 : 1.0 - (1.0 - h1)/(1.0 - h0) # mortality
end

"""
Hazard Calculation for Age-Dependent Process

The hazard is computed as ``\\frac{F(x,k,θ) - F(x-1,k,θ)}{1 - F(x-1,k,θ)} ``

"""
function age_hazard_calc(heval::Function, d::Number, q::Number, k::Number, theta::Number, qkey::Tuple)
    h0 = q == zero(q) ? 0.0 : heval(q - one(q), k, theta)
    h1 = heval(q, k, theta)
    h0 == 1.0 ? 1.0 : 1.0 - (1.0 - h1)/(1.0 - h0) # mortality
end

"""
Hazard Calculation for Constant-Probability Process

"""
function age_const_calc(heval::Function, d::Number, q::Number, k::Number, theta::Number, qkey::Tuple)
    Float64(theta)
end

function age_dummy_calc(heval::Function, d::Number, q::Number, k::Number, theta::Number, qkey::Tuple)
    Float64(0.0)
end

function age_hazard_check(a::Int64)
    return false
end

function age_custom_check(a::Int64)
    return false
end

function acc_hazard_check(d::Float64)
    return d >= ACCTHR
end

# --------------------------------------------------------------------------------
# stepper type
# --------------------------------------------------------------------------------

"""
Key for population development tables

A struct containing `age` (integer) and development fraction `dev` (float).

"""
abstract type StepperTypes end
abstract type AccStep <: StepperTypes end
abstract type AgeStep <: StepperTypes end
abstract type CusStep <: StepperTypes end

struct AccStepper <: AccStep
    step::Float64
    function AccStepper()
        new(0.0)
    end
    function AccStepper(q::Number, d::Int64, k::Number)
        new(d == 0 ? q : round(q + Float64(d)/Float64(k), digits=EPS))
    end
end

struct AgeStepper <: AgeStep
    step::Int64
    function AgeStepper()
        new(0)
    end
    function AgeStepper(q::Number, d::Int64, k::Number)
        new(q + one(q))
    end
end

struct CustomStepper <: CusStep
    step::Int64
    function CustomStepper()
        new(0)
    end
    function CustomStepper(q::Number, d::Int64, k::Number)
        new(q)
    end
end

# accumulation types ------------------------------------------------------------

# fixed accumulation
function acc_fixed_pars(pars::T) where {T <: NamedTuple}
    k = round(pars.devmn)
    theta = 1.0
    return k, theta, true
end

function acc_fixed_haz(i::Number, k::Number, theta::Number)
    Float64(i >= theta)
end

"""
Fixed Duration Accumulative Development Process

This accumulative development process employs a step function, with discontunity at `devmn`, as the cumulative density function.

`AccFixed()` returns a struct with fields:
- `pars` takes arguments `devmn` and `devsd` which computes `k`, `theta` (returned as a tuple in that order)
- `eval` takes arguments `i` and `theta` and returns the cumulative density function evaluated at `i`
- `func` takes arguments `age`, `dev`, `hazard::AccHaz`, `k`, and `theta` and returns the hazard evaluated at `dev`

The struct `AccFixed` inherits from the abstract type `AccHaz` (which itself has supertype `HazTypes`).

"""
struct AccFixed <: AccHaz
    pars::Function
    eval::Function
    func::Function
    check::Function
    stepper::Type{S} where {S <: StepperTypes}
    function AccFixed()
        new(acc_fixed_pars, acc_fixed_haz, acc_hazard_calc, acc_hazard_check, AccStepper)
    end
end

# pascal
function acc_pascal_pars(pars::T) where {T <: NamedTuple}
    theta = pars.devmn / (pars.devsd^2)
    (theta < 1.0 && theta > 0.0) || throw(ArgumentError("Pascal cannot yield mean=$(pars.devmn) and sd=$(pars.devsd)"))
    k = pars.devmn * theta / (1.0 - theta)
    if k != round(k)
        k = round(k)
        theta = k / (pars.devmn + k)
    end
    return k, theta, true
end

function acc_pascal_haz(i::Number, k::Number, theta::Number)
    Float64(1.0 - theta^(i + 1.0))
end

"""
Pascal Accumulative Development Process

This Pascal (or negative binomial) development process employs the negative binomial distribution to represent process duration.

`AccPascal()` returns a struct with fields:
- `pars` takes arguments `devmn` and `devsd` which computes `k`, `theta` (returned as a tuple in that order)
- `eval` takes arguments `i` and `theta` and returns the cumulative density function evaluated at `i`
- `func` takes arguments `age`, `dev`, `hazard::AccHaz`, `k`, and `theta` and returns the hazard evaluated at `dev`
    
The struct `AccFixed` inherits from the abstract type `AccHaz` (which itself has supertype `HazTypes`).

"""
struct AccPascal <: AccHaz
    pars::Function
    eval::Function
    func::Function
    check::Function
    stepper::Type{S} where {S <: StepperTypes}
    function AccPascal()
        new(acc_pascal_pars, acc_pascal_haz, acc_hazard_calc, acc_hazard_check,AccStepper)
    end
end

# Erlang
function acc_erlang_pars(pars::T) where {T <: NamedTuple}
    theta = (pars.devsd^2) / pars.devmn
    k = pars.devmn / theta
    if k != round(k)
        k = round(k)
        theta = pars.devmn / k
        m = k*theta
        s = (theta*m)^0.5
        # @error string("Rounding up k to ", k, " to yield mean=", m, " and sd=", s)
    end
    return k, theta, true
end

function acc_erlang_haz(i::Number, k::Number, theta::Number)
    cdf(Poisson(1.0/theta), i)
end

"""
Erlang Accumulative Development Process

The Erlang development process uses the Gamma distribution with an integer shape parameter to represent process duration.

`AccPascal()` returns a struct with fields:
- `pars` takes arguments `devmn` and `devsd` which computes `k`, `theta` (returned as a tuple in that order)
- `eval` takes arguments `i` and `theta` and returns the cumulative density function evaluated at `i`
- `func` takes arguments `age`, `dev`, `hazard::AccHaz`, `k`, and `theta` and returns the hazard evaluated at `dev`
    
The struct `AccFixed` inherits from the abstract type `AccHaz` (which itself has supertype `HazTypes`).

"""
struct AccErlang <: AccHaz
    pars::Function
    eval::Function
    func::Function
    check::Function
    stepper::Type{S} where {S <: StepperTypes}
    function AccErlang()
        new(acc_erlang_pars, acc_erlang_haz, acc_hazard_calc, acc_hazard_check, AccStepper)
    end
end

# age types ------------------------------------------------------------

# constant age
function age_const_pars(pars::T) where {T <: NamedTuple}
    k = 1.0
    theta = min(1.0, max(0.0, pars.prob))
    return k, theta, false
end

function age_const_haz(i::Number, k::Number, theta::Number)
    Float64(theta)
end

"""
Constant Probability Age-Dependent Development Process

This age-dependent development process employs a constant probability of occurrence per step.

`AgeConst()` returns a struct with fields:
- `pars` takes arguments `devmn` and `devsd` which computes `k`, `theta` (returned as a tuple in that order)
- `eval` takes arguments `i` and `theta` and returns the cumulative density function evaluated at `i`
- `func` takes arguments `age`, `dev`, `hazard::AgeHaz`, `k`, and `theta` and returns the hazard evaluated at `dev`

The struct `AgeConst` inherits from the abstract type `AgeHaz` (which itself has supertype `HazTypes`).

"""
struct AgeConst <: AgeHaz
    pars::Function
    eval::Function
    func::Function
    check::Function
    stepper::Type{S} where {S <: StepperTypes}
    function AgeConst()
        new(age_const_pars, age_const_haz, age_const_calc, age_hazard_check, AgeStepper)
    end
end

function age_custom_pars(pars::T) where {T <: NamedTuple}
    return 1, 1.0, false
end

function age_custom_haz(i::Number, k::Number, theta::Number)
    return nothing
end

"""
Custom Probability Age-Dependent Development Process

This age-dependent development process employs a custom probability of occurrence per step.

`AgeConst()` returns a struct with fields:
- `pars` takes arguments `devmn` and `devsd` which computes `k`, `theta` (returned as a tuple in that order)
- `eval` takes arguments `i` and `theta` and returns the cumulative density function evaluated at `i`
- `func` takes arguments `age`, `dev`, `hazard::AgeHaz`, `k`, and `theta` and returns the hazard evaluated at `dev`

The struct `AgeConst` inherits from the abstract type `AgeHaz` (which itself has supertype `HazTypes`).

"""
struct AgeCustom <: CusHaz
    pars::Function
    eval::Function
    func::Function
    check::Function
    stepper::Type{S} where {S <: StepperTypes}
    function AgeCustom(age_custom_calc::Function)
        new(age_custom_pars, age_custom_haz, age_custom_calc, age_custom_check, CustomStepper)
    end
    function AgeCustom(age_custom_calc::Function, stepper::Type{S}) where {S <: StepperTypes}
        new(age_custom_pars, age_custom_haz, age_custom_calc, age_custom_check, stepper)
    end
end

struct AgeDummy <: CusHaz
    pars::Function
    eval::Function
    func::Function
    check::Function
    stepper::Type{S} where {S <: StepperTypes}
    function AgeDummy()
        new(age_custom_pars, age_custom_haz, age_dummy_calc, age_custom_check, CustomStepper)
    end
end

# fixed age
function age_fixed_pars(pars::T) where {T <: NamedTuple}
    k = round(pars.devmn)
    theta = 1.0
    return k, theta, false
end

function age_fixed_haz(i::Number, k::Number, theta::Number)
    Float64(i >= k)
end

"""
Fixed Duration Age-Dependent Development Process

This age-dependent development process employs a step function, with discontunity at `devmn`, as the cumulative density function.

`AgeFixed()` returns a struct with fields:
- `pars` takes arguments `devmn` and `devsd` which computes `k`, `theta` (returned as a tuple in that order)
- `eval` takes arguments `i` and `theta` and returns the cumulative density function evaluated at `i`
- `func` takes arguments `age`, `dev`, `hazard::AgeHaz`, `k`, and `theta` and returns the hazard evaluated at `dev`

The struct `AgeFixed` inherits from the abstract type `AgeHaz` (which itself has supertype `HazTypes`).

"""
struct AgeFixed <: AgeHaz
    pars::Function
    eval::Function
    func::Function
    check::Function
    stepper::Type{S} where {S <: StepperTypes}
    function AgeFixed()
        new(age_fixed_pars, age_fixed_haz, age_hazard_calc, age_hazard_check, AgeStepper)
    end
end

# negative binomial age
function age_nbinom_pars(pars::T) where {T <: NamedTuple}
    theta = pars.devmn / (pars.devsd^2)
    (theta < 1.0 && theta > 0.0) || throw(ArgumentError("Negative binomial cannot yield mean=$(pars.devmn) and sd=$(pars.devsd)"))
    k = pars.devmn * theta / (1.0 - theta)
    return k, theta, false
end

function age_nbinom_haz(i::Number, k::Number, theta::Number)
    cdf(NegativeBinomial(k, theta), i - 1)
end

"""
Negative Binomial Age-Dependent Development Process

The duration of this age-dependent development process follows a negative binomial distribution.

`AgeNbinom()` returns a struct with fields:
- `pars` takes arguments `devmn` and `devsd` which computes `k`, `theta` (returned as a tuple in that order)
- `eval` takes arguments `i` and `theta` and returns the cumulative density function evaluated at `i`
- `func` takes arguments `age`, `dev`, `hazard::AgeHaz`, `k`, and `theta` and returns the hazard evaluated at `dev`

The struct `AgeNbinom` inherits from the abstract type `AgeHaz` (which itself has supertype `HazTypes`).

"""
struct AgeNbinom <: AgeHaz
    pars::Function
    eval::Function
    func::Function
    check::Function
    stepper::Type{S} where {S <: StepperTypes}
    function AgeNbinom()
        new(age_nbinom_pars, age_nbinom_haz, age_hazard_calc, age_hazard_check, AgeStepper)
    end
end

# gamma age
function age_gamma_pars(pars::T) where {T <: NamedTuple}
    theta = (pars.devsd^2) / pars.devmn
    k = pars.devmn / theta
    return k, theta, false
end

function age_gamma_haz(i::Number, k::Number, theta::Number)
    cdf(Gamma(k, theta), i)
end

"""
Gamma Age-Dependent Development Process

This age-dependent development process follows a gamma distribution.

`AgeGamma()` returns a struct with fields:
- `pars` takes arguments `devmn` and `devsd` which computes `k`, `theta` (returned as a tuple in that order)
- `eval` takes arguments `i` and `theta` and returns the cumulative density function evaluated at `i`
- `func` takes arguments `age`, `dev`, `hazard::AgeHaz`, `k`, and `theta` and returns the hazard evaluated at `dev`

The struct `AgeGamma` inherits from the abstract type `AgeHaz` (which itself has supertype `HazTypes`).

"""
struct AgeGamma <: AgeHaz
    pars::Function
    eval::Function
    func::Function
    check::Function
    stepper::Type{S} where {S <: StepperTypes}
    function AgeGamma()
        new(age_gamma_pars, age_gamma_haz, age_hazard_calc, age_hazard_check, AgeStepper)
    end
end


# --------------------------------------------------------------------------------
# Population data types
# --------------------------------------------------------------------------------

abstract type PopDataTypes end

# combined age- and acumulated-development Population members -------------------

struct MemberKey
    key::Tuple{Number,Number,Number,Number,Number}
    function MemberKey(n1::Number)
        return new((n1,-1,-1,-1,-1))
    end
    function MemberKey(n1::Number,n2::Number)
        return new((n1,n2,-1,-1,-1))
    end
    function MemberKey(n1::Number,n2::Number,n3::Number)
        return new((n1,n2,n3,-1,-1))
    end
    function MemberKey(n1::Number,n2::Number,n3::Number,n4::Number)
        return new((n1,n2,n3,n4,-1))
    end
    function MemberKey(n1::Number,n2::Number,n3::Number,n4::Number,n5::Number)
        return new((n1,n2,n3,n4,n5))
    end
    function MemberKey(h::Array{HazTypes, 1})
        n = length(h)
        if n > 5
            throw(ArgumentError("At most 5 processes are allowed"))
        end
        return new((n>0 ? h[1].stepper().step : -1,
                    n>1 ? h[2].stepper().step : -1,
                    n>2 ? h[3].stepper().step : -1,
                    n>3 ? h[4].stepper().step : -1,
                    n>4 ? h[5].stepper().step : -1))
    end
    function MemberKey(m::MemberKey, h::Array{HazTypes, 1}, n::Int64, d::Int64, k::Number)
        if n > 5 || n < 1
            throw(ArgumentError("At most 5 processes are allowed"))
        end
        return new(Tuple([n==i ? h[i].stepper(m.key[i],d,k).step : m.key[i] for i in 1:5]))
    end
end

"""
Population data for deterministic models

Return a struct inheriting from `PopDataTypes` with 3 fields:
- `poptable`: a `Dict` mapping `Float64` by `MemberKey` keys

"""
struct PopDataDet <: PopDataTypes
    poptable::Dict{MemberKey, Float64}
    function PopDataDet()
        new(Dict{MemberKey, Float64}())
    end
end

"""
Population data for stochastic models

Return a struct inheriting from `PopDataTypes` with 3 fields:
- `poptable`: a `Dict` mapping `Int64` by `MemberKey` keys

"""
struct PopDataSto <: PopDataTypes
    poptable::Dict{MemberKey, Int64}
    function PopDataSto()
        new(Dict{MemberKey, Int64}())
    end
end

"""
Add key-value pair to development table

Add the key `key` and value `n` to the development table `data`. If the key already exists, `n` is added to its value, otherwise a new entry is created.

"""
function add_key(data::Dict{MemberKey, T}, key::MemberKey, n::T) where {T<:Number}
    if haskey(data, key)
        data[key] += n
    else
        data[key] = n
    end
end

"""
Tabulate development table

Return a tuple of two `Dict` objects, the first which indexes counts by development ages and the second by fraction of development completed.

"""
function GetPoptable(poptable::Dict{M, T}) where {T<:Number, M<:MemberKey}
    r = []
    for (x,n) in poptable
        for i in 1:5
            if x.key[i] >= 0
                if length(r)<i; push!(r,Dict()); end
                if haskey(r[i], x.key[i]); r[i][x.key[i]] += n; else; r[i][x.key[i]] = n; end
            end
        end
    end
    return r
end

# --------------------------------------------------------------------------------
# update type
# --------------------------------------------------------------------------------

abstract type UpdateTypes end

"""
Stochastic update

A callable struct inheriting from `UpdateTypes` which draws a binomial random variate.

"""
struct StochasticUpdate <: UpdateTypes end

"""
Deterministic update

A callable struct inheriting from `UpdateTypes` which returns the proportion experiencing an event.

"""
struct DeterministicUpdate <: UpdateTypes end

# stochastic
function (::StochasticUpdate)(n, p)
    rand(Binomial(n, p))
end

# deterministic
function (::DeterministicUpdate)(n, p)
    n*p
end


# --------------------------------------------------------------------------------
# Population struct
# --------------------------------------------------------------------------------

"""
A population

A struct containing a single population. It can be constructed by passing two arguments to its constructor,
`d` should be either `PopDataSto` or `PopDataDet` and `h` is the hazard type, and should inherit from `AgeHaz` or `AccHaz`,
for ageing or accumulation based development processes, respectively.

"""
struct Population
    data::PopDataTypes
    update::UpdateTypes
    hazards::Array{HazTypes, 1}
    function Population(d::PopDataTypes)
        u::UpdateTypes = typeof(d) <: PopDataDet ? DeterministicUpdate() : StochasticUpdate()
        new(d, u, [])
    end
end

"""
Add processes in the order to be executed

"""
function AddProcess(pop::Population, h1::HazTypes)
    push!(pop.hazards, h1)
end

function AddProcess(pop::Population, h1::HazTypes, h2::HazTypes)
    for h in (h1, h2)
        push!(pop.hazards, h)
    end
end

function AddProcess(pop::Population, h1::HazTypes, h2::HazTypes, h3::HazTypes)
    for h in (h1, h2, h3)
        push!(pop.hazards, h)
    end
end

function AddProcess(pop::Population, h1::HazTypes, h2::HazTypes, h3::HazTypes, h4::HazTypes)
    for h in (h1, h2, h3, h4)
        push!(pop.hazards, h)
    end
end

function AddProcess(pop::Population, h1::HazTypes, h2::HazTypes, h3::HazTypes, h4::HazTypes, h5::HazTypes)
    for h in (h1, h2, h3, h4, h5)
        push!(pop.hazards, h)
    end
end

"""
Add individuals to a population

Add individuals to an existing population. Individuals can be added by either passing the number, age, and development fraction attained, or
by passing a second `Population` object which will be added to the first. The function also allows a custom development cycle,
i.e. the number of times an individual completed the development process.

"""
function AddPop(pop::Population, n::Number)
    key = MemberKey(pop.hazards)
    add_key(pop.data.poptable, key, n)
end

function AddPop(pop::Population, n::Number, h1::Number)
    key = MemberKey(h1)
    add_key(pop.data.poptable, key, n)
end

function AddPop(pop::Population, n::Number, h1::Number, h2::Number)
    key = MemberKey(h1,h2)
    add_key(pop.data.poptable, key, n)
end

function AddPop(pop::Population, n::Number, h1::Number, h2::Number, h3::Number)
    key = MemberKey(h1,h2,h3)
    add_key(pop.data.poptable, key, n)
end

function AddPop(pop::Population, n::Number, h1::Number, h2::Number, h3::Number, h4::Number)
    key = MemberKey(h1,h2,h3,h4)
    add_key(pop.data.poptable, key, n)
end

function AddPop(pop::Population, n::Number, h1::Number, h2::Number, h3::Number, h4::Number, h5::Number)
    key = MemberKey(h1,h2,h3,h4,h5)
    add_key(pop.data.poptable, key, n)
end

function AddPop(popto::Population, popfrom::Population)
    for (q,n) in popfrom.data.poptable
        add_key(popto.data.poptable, q, n)
    end
end

"""
Get size of a population

Return the total number of individuals in this population.

"""
function GetPop(pop::Population)
    size = zero(valtype(pop.data.poptable))
    for n in values(pop.data.poptable)
        size += n
    end
    return size
end

# --------------------------------------------------------------------------------
# renew a Population
# --------------------------------------------------------------------------------

"""
Empty a population

Remove all individuals from this population.

"""
function EmptyPop(pop::Population)
    empty!(pop.data.poptable)
    #
    return true
end

# --------------------------------------------------------------------------------
# step function
# --------------------------------------------------------------------------------

"""
Iterate a population

Update a population over a single time step, `devmn` is the current mean number of time steps until development completes, `devsd` is
its standard deviation, and `death` is the per-capita mortality probability.

"""
function StepPopMain(pop::Population, pars::Tuple)
    completed = [zero(valtype(pop.data.poptable)) for _ in 1:length(pop.hazards)]
    #
    hazpar = []
    for i in 1:length(pop.hazards)
        k, theta, stay = pop.hazards[i].pars(pars[i])
        push!(hazpar, (k=k, theta=theta, stay=stay))
    end
    #
    poptable = pop.data.poptable
    poptablenext = Dict{keytype(poptable),valtype(poptable)}()
    poptabledone = [Dict{keytype(poptable),valtype(poptable)}() for _ in 1:length(pop.hazards)]
    #
    for i in 1:length(pop.hazards)
        hazard = pop.hazards[i]
        k, theta, stay = hazpar[i]
        if theta == 0.0 || k == 0
            continue
        end
        #
        for (q,n) in poptable
            if n == zero(valtype(pop.data.poptable))
                continue
            end
            #
            dev = zero(Int64)
            while n > zero(valtype(pop.data.poptable))
                q2 = MemberKey(q, pop.hazards, i, dev, k)
                #
                if hazard.check(q2.key[i])
                    add_key(poptabledone,q2,n)
                    completed[i] += n
                    n = zero(valtype(pop.data.poptable))
                else
                    p = hazard.func(hazard.eval, dev, q2.key[i], k, theta, q2.key)
                    n2 = pop.update(n, p)
                    n -= n2
                    #
                    if typeof(hazard) <: AgeHaz || typeof(hazard) <: CusHaz
                        if n > zero(valtype(pop.data.poptable))
                            add_key(poptablenext, q2, n) # Developing / surviving population
                        end
                        if n2 > zero(valtype(pop.data.poptable))
                            add_key(poptabledone[i], q2, n2) # Completing process
                            completed[i] += n2
                        end
                    else
                        if n2 > zero(valtype(pop.data.poptable))
                            add_key(poptablenext, q2, n2) # Developing / surviving population
                        end
                    end
                    #
                    dev += one(dev)
                end
                #
                if !stay
                    break
                end
            end
        end
        #
        poptable = poptablenext
        poptablenext = Dict{keytype(poptable),valtype(poptable)}()
    end
    size = zero(valtype(pop.data.poptable))
    empty!(pop.data.poptable)
    for (q,n) in poptable
        pop.data.poptable[q] = n
        size += n
    end
    return size, completed, poptabledone    
end

function StepPop(pop::Population, pr1::NamedTuple)
    StepPopMain(pop, (pr1,))
end

function StepPop(pop::Population, pr1::NamedTuple, pr2::NamedTuple)
    StepPopMain(pop, (pr1, pr2))
end

function StepPop(pop::Population, pr1::NamedTuple, pr2::NamedTuple, pr3::NamedTuple)
    StepPopMain(pop, (pr1, pr2, pr3))
end

function StepPop(pop::Population, pr1::NamedTuple, pr2::NamedTuple, pr3::NamedTuple, pr4::NamedTuple)
    StepPopMain(pop, (pr1, pr2, pr3, pr4))
end

function StepPop(pop::Population, pr1::NamedTuple, pr2::NamedTuple, pr3::NamedTuple, pr4::NamedTuple, pr5::NamedTuple)
    StepPopMain(pop, (pr1, pr2, pr3, pr4, pr5))
end

end
