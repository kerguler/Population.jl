#=
sPop2: a dynamically-structured matrix population model
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

export acc_haz, age_haz, haz_types,
        acc_fixed, acc_pascal, acc_erlang,
        age_fixed, age_nbinom, age_gamma, 
        pop_data_det, pop_data_sto,
        population, step_pop, add_pop, get_pop,
        set_eps, empty_pop, get_poptable

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

abstract type haz_types end
abstract type acc_haz <: haz_types end
abstract type age_haz <: haz_types end

function acc_hazard_calc(age::Number, dev::Number, hazard::acc_haz, k::Number, theta::Number)
    h0 = dev == 0 ? 0.0 : hazard.eval(dev - 1, theta)
    h1 = hazard.eval(dev, theta)
    h0 == 1.0 ? 1.0 : (h1 - h0) / (1.0 - h0)
end

function age_hazard_calc(age::Number, dev::Number, hazard::age_haz, k::Number, theta::Number)
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

struct acc_fixed <: acc_haz
    pars::Function
    eval::Function
    func::Function
    function acc_fixed()
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

struct acc_pascal <: acc_haz
    pars::Function
    eval::Function
    func::Function
    function acc_pascal()
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

struct acc_erlang <: acc_haz
    pars::Function
    eval::Function
    func::Function
    function acc_erlang()
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

struct age_fixed <: age_haz
    pars::Function
    eval::Function
    func::Function
    function age_fixed()
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

struct age_nbinom <: age_haz
    pars::Function
    eval::Function
    func::Function
    function age_nbinom()
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

struct age_gamma <: age_haz
    pars::Function
    eval::Function
    func::Function
    function age_gamma()
        new(age_gamma_pars, age_gamma_haz, age_hazard_calc)
    end
end


# --------------------------------------------------------------------------------
# population data types
# --------------------------------------------------------------------------------

abstract type pop_data_types end
abstract type sto_data <: pop_data_types end
abstract type det_data <: pop_data_types end

# combined age- and acumulated-development population members -------------------

struct member_key
    age::Int64
    dev::Float64
    function member_key(a::Int64, d::Float64)
        new(a, round(d, digits=EPS))
    end
end

function add_key(data::Dict{member_key, Float64}, key::member_key, n::Float64)
    if haskey(data, key)
        data[key] += n
    else
        data[key] = n
    end
end

function add_key(data::Dict{member_key, Int64}, key::member_key, n::Int64)
    if haskey(data, key)
        data[key] += n
    else
        data[key] = n
    end
end

# deterministic
struct pop_data_det <: det_data
    poptable_current::Dict{member_key, Float64}
    poptable_next::Dict{member_key, Float64}
    poptable_done::Dict{member_key, Float64}
    function pop_data_det()
        new(Dict{member_key, Float64}(),Dict{member_key, Float64}(),Dict{member_key, Float64}())
    end
end

function get_poptable(poptable::Dict{member_key, Float64})
    ra = Dict{Int64, Float64}()
    rd = Dict{Float64, Float64}()
    for (x,n) in poptable
        ra[x.age] = haskey(ra,x.age) ? ra[x.age] + n : n
        rd[x.dev] = haskey(rd,x.dev) ? rd[x.dev] + n : n
    end
    return ra, rd
end

# stochastic
struct pop_data_sto <: sto_data
    poptable_current::Dict{member_key, Int64}
    poptable_next::Dict{member_key, Int64}
    poptable_done::Dict{member_key, Int64}
    function pop_data_sto()
        new(Dict{member_key, Int64}(),Dict{member_key, Int64}(),Dict{member_key, Int64}())
    end
end

function get_poptable(poptable::Dict{member_key, Int64})
    ra = Dict{Int64, Int64}()
    rd = Dict{Float64, Int64}()
    for (x,n) in poptable
        add_key(ra, x.age, n)
        add_key(rd, x.dev, n)
    end
    return ra, rd
end

# --------------------------------------------------------------------------------
# update type
# --------------------------------------------------------------------------------

abstract type update_types end
struct stochastic_update <: update_types end
struct deterministic_update <: update_types end

# stochastic
function (::stochastic_update)(n, p)
    rand(Binomial(n, p))
end

# deterministic
function (::deterministic_update)(n, p)
    n*p
end


# --------------------------------------------------------------------------------
# population struct
# --------------------------------------------------------------------------------

struct population{T<:pop_data_types,H<:haz_types,F<:update_types}
    data::T
    hazard::H
    update::F
    function population(d::T, h::H) where {T <: pop_data_types, H <: age_haz}
        u::update_types = T <: det_data ? deterministic_update() : stochastic_update()
        new{T,H,update_types}(d, h, u)
    end
    function population(d::T, h::H) where {T <: pop_data_types, H <: acc_haz}
        u::update_types = T <: det_data ? deterministic_update() : stochastic_update()
        new{T,H,update_types}(d, h, u)
    end
end

function add_pop(pop::population{T,H,F}, n::Number, age::Number, dev::Number) where {T<:pop_data_types,H<:haz_types,F<:update_types}
    key = member_key(max(age,0), max(0.0,dev))
    pop.data.poptable_current[key] = n
end

function add_pop(popto::population{T,Ht,Ft}, popfrom::population{T,Hf,Ff}) where {T<:pop_data_types,Ht<:haz_types,Hf<:haz_types,Ft<:update_types,Ff<:update_types}
    for (q,n) in popfrom.data.poptable_current
        if haskey(popto.data.poptable_current, q)
            popto.data.poptable_current[q] += n
        else 
            popto.data.poptable_current[q] = n
        end
    end
end

function get_pop(pop::population{T,H,F}) where {T<:pop_data_types,H,F}
    size = zero(valtype(pop.data.poptable_current))
    for n in values(pop.data.poptable_current)
        size += n
    end
    return size
end

# --------------------------------------------------------------------------------
# renew a population
# --------------------------------------------------------------------------------

function empty_pop(pop::population{T,H,F}) where {T<:pop_data_types,H<:haz_types,F<:update_types}
    empty!(pop.data.poptable_current)
    empty!(pop.data.poptable_next)
    empty!(pop.data.poptable_done)
    #
    return true
end

# --------------------------------------------------------------------------------
# step function
# --------------------------------------------------------------------------------

function step_pop(pop::population{T,H,F}, devmn::Number, devsd::Number, death::Number) where {T<:pop_data_types,H<:haz_types,F<:update_types}
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
        # age
        age = q.age + one(q.age)
        # mortality
        if death > 0.0
            dd = pop.update(n, death)
            dead += dd
            n -= dd
        end
        # development
        if theta == 0.0 || k == 0
            q2 = member_key(age, q.dev)
            pop.data.poptable_next[q2] = n
            continue
        end
        #
        dev = 0
        while n > zero(valtype(pop.data.poptable_current))
            q2 = member_key(age, q.dev + dev/k)
            if q2.dev >= ACCTHR
                add_key(pop.data.poptable_done, q2, n)
                developed += n
                n = zero(valtype(pop.data.poptable_current))
            else
                p = pop.hazard.func(age, dev, pop.hazard, k, theta)
                n2 = pop.update(n, p)
                n -= n2
                #
                if typeof(pop.hazard) <: age_haz
                    if n2 > zero(valtype(pop.data.poptable_current))
                        add_key(pop.data.poptable_done, q2, n2)
                        developed += n2
                    end
                    add_key(pop.data.poptable_next, q2, n)
                    break
                else
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