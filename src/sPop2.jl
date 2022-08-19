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

export newPop,
       addPop,
       mergePop,
       stepPop,
       getHtk,
       readKeys

using Distributions
using Printf

EPS = 14
ACCTHR = 1.0

GAMMA_MODES = Set(["ACC_ERLANG",
                   "ACC_PASCAL",
                   "ACC_FIXED",
                   "AGE_FIXED",
                   "AGE_GAMMA",
                   "AGE_NBINOM"])

mutable struct Qnit
    age::Int64
    dev::Float64
end

mutable struct Pop
    stochastic::Bool
    gammafun::String
    devc::Dict
    size::Union{Int64,Float64}
end

function addKey(devc::Dict,
                q::Qnit,
                n::Union{Int64,Float64})
    x = (q.age, q.dev)
    devc[x] = haskey(devc, x) ? devc[x] + n : n
end

function readKey(x)
    return Qnit(x[1], x[2])
end

function readKeys(devc::Dict)
    ra = Dict()
    rd = Dict()
    for (x,n) in devc
        tmp = readKey(x)
        ra[tmp.age] = haskey(ra,tmp.age) ? ra[tmp.age] + n : n
        rd[tmp.dev] = haskey(rd,tmp.dev) ? rd[tmp.dev] + n : n
    end
    return ra, rd
end

# Age-dependent
function HFixed(i::Int64,k::Int64,theta::Float64);    convert(Float64, i >= k);              end
function HNBinom(i::Int64,k::Float64,theta::Float64); cdf(NegativeBinomial(k,theta), i - 1); end
function HGamma(i::Int64,k::Float64,theta::Float64);  cdf(Gamma(k,theta), i);                end
# Accumulating
function HFixedACC(i::Int64,theta::Float64); convert(Float64, i >= theta); end
function HPascal(i::Int64,theta::Float64);   1.0 - theta^(i + 1.0);        end
function HErlang(i::Int64,theta::Float64);   cdf(Poisson(1.0/theta), i);   end
# Hash (under construction)
hDistribution = Dict()
function hHErlang(i::Int64,k::Float64,theta::Float64)
    label = "ACC_ERLANG"
    key = (k, theta, i)
    if !haskey(label); hDistribution[label] = Dict(); end
    if !haskey(hDistribution[label], key); hDistribution[label][key] = HErlang(i,k,theta); end
    return hDistribution[label][key]
end

function getHtk(gammafun::String,
                devmn::Float64,
                devsd::Float64,
                verbose::Bool = false)
    if devmn == 0.0 && devsd == 0.0
        return Dict("H"=>HFixed, "theta"=>0.0, "k"=>0, "acc"=>false)
    end
    #
    makeint::Bool = true
    theta::Float64 = 0.0
    k::Float64 = 0.0
    acc::Bool = false
    if gammafun === "AGE_FIXED"
        H = HFixed
        k = round(devmn)
        theta = 1.0
    elseif gammafun === "AGE_GAMMA"
        H = HGamma
        theta = devsd * devsd / devmn
        k = devmn / theta
        makeint = false
    elseif gammafun === "AGE_NBINOM"
        H = HNBinom
        theta = devmn / (devsd * devsd)
        if theta >= 1.0 || theta == 0.0
            @error string("Negative binomial cannot yield mean=", devmn, " and sd=", devsd)
            return false
        end
        k = devmn * theta / (1.0 - theta)
        makeint = false
    elseif gammafun === "ACC_FIXED"
        H = HFixedACC
        k = round(devmn)
        theta = 1.0
        acc = true
    elseif gammafun === "ACC_ERLANG"
        H = HErlang
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
        acc = true
    elseif gammafun === "ACC_PASCAL"
        H = HPascal
        theta = devmn / (devsd * devsd)
        if theta >= 1.0 || theta == 0.0
            @error string("Pascal cannot yield mean=", devmn, " and sd=", devsd)
            return false
        end
        k = devmn * theta / (1.0 - theta)
        if k != round(k)
            k = round(k)
            theta = k / (devmn + k)
        end
        acc = true
    else
        @error string("The distribution id=", gammafun, " is not supported\nSupported distributions are ", string(GAMMA_MODES))
        return false
    end
    #
    if makeint
        kk::Int64 = convert(Int64, round(k))
        if kk == 0
            @error string("In accumulated development, the number of pseudo-states should be at least 1 (k=", k,")")
            return false
        end
        #
        return Dict("H"=>H, "theta"=>theta, "k"=>kk, "acc"=>acc)
    end
    return Dict("H"=>H, "theta"=>theta, "k"=>k, "acc"=>acc)
end

function newPop(stochastic::Bool=false,
                gammafun::String="ACC_ERLANG")
    if !(gammafun in GAMMA_MODES)
        @error string("The distribution id=", gammafun, " is not supported\nSupported distributions are ", string(GAMMA_MODES))
        return false
    end
    #
    mty = Dict()
    Pop(stochastic, 
        gammafun,
        mty,
        stochastic ? 0 : 0.0)
end

function addPop(pop::Pop,
                age::Int64,
                dev::Float64,
                num::Union{Int64,Float64})
    dev = round(dev, digits=EPS)
    qnit = Qnit(age,dev)
    addKey(pop.devc, qnit, num)
    pop.size += num
    #
    return true
end

function mergePop(pop::Pop,
                  addpop::Pop)
    if pop.stochastic != addpop.stochastic
        @error "The two populations are not compatible"
        return false
    end
    #
    for (q,n) in addpop.devc
        pop.devc[q] = haskey(pop.devc, q) ? pop.devc[q] + n : n
        pop.size += n
    end
    #
    return true
end

function stepPop(pop::Pop,
                 devmn::Float64,
                 devsd::Float64,
                 death::Float64)
    # Obtain development time Hazard function and the associated parameters
    tmp = getHtk(pop.gammafun, devmn, devsd)
    if tmp == false; return false; end
    H = tmp["H"]
    theta = tmp["theta"]
    k = tmp["k"]
    acc = tmp["acc"]
    # Keep track of developed and dead
    developed::Union{Int64,Float64} = pop.stochastic ? 0 : 0.0
    dead::Union{Int64,Float64} = pop.stochastic ? 0 : 0.0
    #
    h0::Float64 = 0.0
    h1::Float64 = 0.0
    p::Float64 = 0.0
    n2::Union{Int64,Float64} = 0.0
    #
    npop = Dict()
    devtable = newPop(pop.stochastic, pop.gammafun)
    for (q,n) in pop.devc
        q = readKey(q)
        # Ageing
        q.age += 1
        # Mortality
        if death > 0.0
            dd::Union{Int64,Float64} = pop.stochastic ? rand(Binomial(n,death)) : n*death
            dead += dd
            n -= dd
        end
        # Development
        if theta > 0.0 && k > 0
            dev::Int64 = 0
            while n > 0.0
                q2 = Qnit(q.age, round(q.dev + dev/k, digits=EPS))
                if acc
                    # Accumulated
                    if q2.dev >= ACCTHR
                        addPop(devtable, q2.age, q2.dev, n)
                        developed += n
                        n = pop.stochastic ? 0 : 0.0
                    else
                        h0 = dev == 0 ? 0.0 : H(dev - 1, theta)
                        h1 = H(dev, theta)
                        p = h0 == 1.0 ? 1.0 : (h1 - h0) / (1.0 - h0)
                        #
                        n2 = pop.stochastic ? rand(Binomial(n,p)) : n*p
                        #
                        addKey(npop, q2, n2)
                        n -= n2
                        #
                        dev += 1
                    end
                else
                    # Age-dependent
                    h0 = H(q2.age - 1, k, theta)
                    h1 = H(q2.age, k, theta)
                    p = h0 == 1.0 ? 1.0 : 1.0 - (1.0 - h1)/(1.0 - h0)
                    #
                    n2 = pop.stochastic ? rand(Binomial(n,p)) : n*p
                    #
                    developed += n2
                    addPop(devtable, q2.age, q2.dev, n2)
                    n -= n2
                    addKey(npop, q2, n)
                    #
                    break
                end
            end
        else
            npop[q] = n
        end
    end
    #
    empty!(pop.devc)
    pop.size = pop.stochastic ? 0 : 0.0
    for (q,n) in npop
        pop.devc[q] = n
        pop.size += n
    end
    #
    return Dict("size"=>pop.size, "developed"=>developed, "dead"=>dead, "devtable"=>devtable)
end

end