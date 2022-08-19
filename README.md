# sPop2: a dynamically-structured matrix population model

<p align="center">
<img width="623" height=211" src="docs/figures/logo_sPop2.jpg"/>
</p>

This is the standalone Julia library of the dynamically-structured matrix population model sPop2. This version implements both age-dependent and accumulated processes.

## Installation

Just type this in Julia:
```julia
using Pkg
Pkg.add("Package Name")
```

## Using the library

The following creates a pseudo-structured population with 10 individuals and iterate it one step with 0 mortality and an Erlang-distributed development time of 20&pm;5 steps.

```julia
pop = newPop(false, "ACC_ERLANG")
addPop(pop, 0, 0.0, 10.0)
stepPop(pop, 20.0, 5.0, 0.0)
```

Please see <a href="docs/">docs/</a> for usage examples.

## References

* Erguler, K., Mendel, J., Petrić, D.V. et al. A dynamically structured matrix population model for insect life histories observed under variable environmental conditions. Sci Rep 12, 11587 (2022). https://doi.org/10.1038/s41598-022-15806-2
* Erguler K. sPop: Age-structured discrete-time population dynamics model in C, Python, and R [version 3; peer review: 2 approved]. F1000Research 2020, 7:1220 (https://doi.org/10.12688/f1000research.15824.3)
