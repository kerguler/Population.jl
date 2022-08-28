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

export AccHaz, AgeHaz, HazTypes,
       AccFixed, AccPascal, AccErlang,
       AgeFixed, AgeNbinom, AgeGamma, 
       PopDataDet, PopDataSto, Population, 
       StepPop, AddPop, GetPop, MemberKey,
       set_eps, EmptyPop, GetPoptable

using Distributions
using Random: rand

const ACCTHR = 1.0

EPS = 14
function set_eps(eps::Int64)
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

"""
Hazard Calculation for Accumulative Development Process

The hazard is computed as ``\\frac{F(x,θ) - F(x-1,θ)}{1 - F(x-1,θ)} ``

"""
function acc_hazard_calc(age::Number, dev::Number, hazard::AccHaz, k::Number, theta::Number)
    h0 = dev == 0 ? 0.0 : hazard.eval(dev - 1, theta)
    h1 = hazard.eval(dev, theta)
    h0 == 1.0 ? 1.0 : (h1 - h0) / (1.0 - h0)
end

"""
Hazard Calculation for Age-Dependent Development Process

The hazard is computed as ``\\frac{F(x,k,θ) - F(x-1,k,θ)}{1 - F(x-1,k,θ)} ``

"""
function age_hazard_calc(age::Number, dev::Number, hazard::AgeHaz, k::Number, theta::Number)
    h0 = hazard.eval(age - 1, k, theta)
    h1 = hazard.eval(age, k, theta)
    h0 == 1.0 ? 1.0 : 1.0 - (1.0 - h1)/(1.0 - h0)
end

# accumulation types ------------------------------------------------------------

# fixed accumulation
function acc_fixed_pars(devmn::Number, devsd::Number)
    k = round(devmn)
    theta = 1.0
    return k, theta
end

function acc_fixed_haz(i::Number, theta::Number)
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
    function AccFixed()
        new(acc_fixed_pars, acc_fixed_haz, acc_hazard_calc)
    end
end

# pascal
function acc_pascal_pars(devmn::Number, devsd::Number)
    theta = devmn / (devsd * devsd)
    (theta < 1.0 && theta > 0.0) || throw(ArgumentError("Pascal cannot yield mean=$(devmn) and sd=$(devsd)"))
    k = devmn * theta / (1.0 - theta)
    if k != round(k)
        k = round(k)
        theta = k / (devmn + k)
    end
    return k, theta
end

function acc_pascal_haz(i::Number, theta::Number)
    1.0 - theta^(i + 1.0)
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
    function AccPascal()
        new(acc_pascal_pars, acc_pascal_haz, acc_hazard_calc)
    end
end

# Erlang
function acc_erlang_pars(devmn::Number, devsd::Number)
    theta = devsd * devsd / devmn
    k = devmn / theta
    if k != round(k)
        k = round(k)
        theta = devmn / k
        m = k*theta
        s = (theta*m)^0.5
        if verbose
            @error string("Rounding up k to ", k, " to yield mean=", m, " and sd=", s)
        end
    end
    return k, theta
end

function acc_erlang_haz(i::Number, theta::Number)
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
    function AccErlang()
        new(acc_erlang_pars, acc_erlang_haz, acc_hazard_calc)
    end
end


# age types ------------------------------------------------------------

# fixed age
function age_fixed_pars(devmn::Number, devsd::Number)
    k = round(devmn)
    theta = 1.0
    return k, theta
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
    function AgeFixed()
        new(age_fixed_pars, age_fixed_haz, age_hazard_calc)
    end
end

# negative binomial age
function age_nbinom_pars(devmn::Number, devsd::Number)
    theta = devmn / (devsd * devsd)
    (theta < 1.0 && theta > 0.0) || throw(ArgumentError("Negative binomial cannot yield mean=$(devmn) and sd=$(devsd)"))
    k = devmn * theta / (1.0 - theta)
    return k, theta
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
    function AgeNbinom()
        new(age_nbinom_pars, age_nbinom_haz, age_hazard_calc)
    end
end

# gamma age
function age_gamma_pars(devmn::Number, devsd::Number)
    theta = devsd * devsd / devmn
    k = devmn / theta
    return k, theta
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
    function AgeGamma()
        new(age_gamma_pars, age_gamma_haz, age_hazard_calc)
    end
end


# --------------------------------------------------------------------------------
# Population data types
# --------------------------------------------------------------------------------

abstract type PopDataTypes end

# combined age- and acumulated-development Population members -------------------

"""
Key for population development tables

A struct containing `age` (integer) and development fraction `dev` (float).

"""
struct MemberKey
    age::Int64
    dev::Float64
    devc::Int64
    function MemberKey(a::Int64, d::Float64, dc::Int64)
        new(a, round(d, digits=EPS), dc)
    end
end

function step_dev(q::MemberKey, d::Int64, k::Number)
    return round(q.dev + Float64(d)/Float64(k), digits=EPS)
end

"""
Population data for deterministic models

Return a struct inheriting from `PopDataTypes` with 3 fields:
- `poptable_current`: a `Dict` mapping `Float64` by `MemberKey` keys
- `poptable_next`: a `Dict` mapping `Float64` by `MemberKey` keys
- `poptable_done`: a `Dict` mapping `Float64` by `MemberKey` keys

"""
struct PopDataDet <: PopDataTypes
    poptable_current::Dict{MemberKey, Float64}
    poptable_next::Dict{MemberKey, Float64}
    poptable_done::Dict{MemberKey, Float64}
    function PopDataDet()
        new(Dict{MemberKey, Float64}(),Dict{MemberKey, Float64}(),Dict{MemberKey, Float64}())
    end
end

"""
Population data for stochastic models

Return a struct inheriting from `PopDataTypes` with 3 fields:
- `poptable_current`: a `Dict` mapping `Int64` by `MemberKey` keys
- `poptable_next`: a `Dict` mapping `Int64` by `MemberKey` keys
- `poptable_done`: a `Dict` mapping `Int64` by `MemberKey` keys

"""
struct PopDataSto <: PopDataTypes
    poptable_current::Dict{MemberKey, Int64}
    poptable_next::Dict{MemberKey, Int64}
    poptable_done::Dict{MemberKey, Int64}
    function PopDataSto()
        new(Dict{MemberKey, Int64}(),Dict{MemberKey, Int64}(),Dict{MemberKey, Int64}())
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
function GetPoptable(poptable::Dict{MemberKey, T}) where {T<:Number}
    ra = Dict{Int64, T}()
    rd = Dict{Float64, T}()
    rdc = Dict{Int64, T}()
    for (x,n) in poptable
        ra[x.age] = haskey(ra,x.age) ? ra[x.age] + n : n
        rd[x.dev] = haskey(rd,x.dev) ? rd[x.dev] + n : n
        rdc[x.devc] = haskey(rd,x.devc) ? rd[x.devc] + n : n
    end
    return ra, rd, rdc
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
struct Population{T<:PopDataTypes,H<:HazTypes,F<:UpdateTypes}
    data::T
    hazard::H
    update::F
    function Population(d::T, h::H) where {T <: PopDataTypes, H <: AgeHaz}
        u::UpdateTypes = T <: PopDataDet ? DeterministicUpdate() : StochasticUpdate()
        new{T,H,UpdateTypes}(d, h, u)
    end
    function Population(d::T, h::H) where {T <: PopDataTypes, H <: AccHaz}
        u::UpdateTypes = T <: PopDataDet ? DeterministicUpdate() : StochasticUpdate()
        new{T,H,UpdateTypes}(d, h, u)
    end
end

"""
Add individuals to a population

Add individuals to an existing population. Individuals can be added by either passing the number, age, and development fraction attained, or
by passing a second `Population` object which will be added to the first. The function also allows a custom development cycle,
i.e. the number of times an individual completed the development process.

"""
function AddPop(pop::Population{T,H,F}, n::Number) where {T<:PopDataTypes,H<:HazTypes,F<:UpdateTypes}
    key = MemberKey(0, 0.0, 0)
    pop.data.poptable_current[key] = n
end

function AddPop(pop::Population{T,H,F}, n::Number, age::Number, dev::Number) where {T<:PopDataTypes,H<:HazTypes,F<:UpdateTypes}
    key = MemberKey(max(age,0), max(0.0,dev), 0)
    pop.data.poptable_current[key] = n
end

function AddPop(pop::Population{T,H,F}, n::Number, age::Number, dev::Number, devc::Number) where {T<:PopDataTypes,H<:HazTypes,F<:UpdateTypes}
    key = MemberKey(max(age,0), max(0.0,dev), max(devc, 0))
    pop.data.poptable_current[key] = n
end

function AddPop(popto::Population{T,Ht,Ft}, popfrom::Population{T,Hf,Ff}) where {T<:PopDataTypes,Ht<:HazTypes,Hf<:HazTypes,Ft<:UpdateTypes,Ff<:UpdateTypes}
    for (q,n) in popfrom.data.poptable_current
        if haskey(popto.data.poptable_current, q)
            popto.data.poptable_current[q] += n
        else 
            popto.data.poptable_current[q] = n
        end
    end
end

"""
Get size of a population

Return the total number of individuals in this population.

"""
function GetPop(pop::Population{T,H,F}) where {T<:PopDataTypes,H,F}
    size = zero(valtype(pop.data.poptable_current))
    for n in values(pop.data.poptable_current)
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
function EmptyPop(pop::Population{T,H,F}) where {T<:PopDataTypes,H<:HazTypes,F<:UpdateTypes}
    empty!(pop.data.poptable_current)
    empty!(pop.data.poptable_next)
    empty!(pop.data.poptable_done)
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
function StepPop(pop::Population{T,H,F}, devmn::Number, devsd::Number, death::Number) where {T<:PopDataTypes,H<:HazTypes,F<:UpdateTypes}
    k, theta = pop.hazard.pars(devmn, devsd)
    dead = zero(valtype(pop.data.poptable_current))
    developed = zero(valtype(pop.data.poptable_current))
    size = zero(valtype(pop.data.poptable_current))
    empty!(pop.data.poptable_done)
    empty!(pop.data.poptable_next)
    for (q,n) in pop.data.poptable_current
        if n == 0
            continue
        end
        # ageing
        age = q.age + one(q.age)
        # mortality
        if death > 0.0
            dd = pop.update(n, death)
            dead += dd
            n -= dd
        end
        # development
        if theta == 0.0 || k == 0
            # skip development
            q2 = MemberKey(age, q.dev, q.devc)
            add_key(pop.data.poptable_next, q2, n)
            continue
        end
        #
        dev = 0
        while n > zero(valtype(pop.data.poptable_current))
            qdev = step_dev(q, dev, k)
            if qdev >= ACCTHR
                q2 = MemberKey(age, qdev, q.devc+1)
                add_key(pop.data.poptable_done, q2, n)
                developed += n
                n = zero(valtype(pop.data.poptable_current))
            else
                p = pop.hazard.func(age, dev, pop.hazard, k, theta)
                n2 = pop.update(n, p)
                n -= n2
                #
                if typeof(pop.hazard) <: AgeHaz
                    if n2 > zero(valtype(pop.data.poptable_current))
                        q2 = MemberKey(age, qdev, q.devc+1)
                        add_key(pop.data.poptable_done, q2, n2)
                        developed += n2
                    end
                    q2 = MemberKey(age, qdev, q.devc)
                    add_key(pop.data.poptable_next, q2, n)
                    break
                else
                    q2 = MemberKey(age, qdev, q.devc)
                    add_key(pop.data.poptable_next, q2, n2)
                end
                dev += 1
            end
        end
    end
    #
    empty!(pop.data.poptable_current)
    for (q,n) in pop.data.poptable_next
        pop.data.poptable_current[q] = n
        size += n
    end
    return size, developed, dead, pop.data.poptable_done
end

end
