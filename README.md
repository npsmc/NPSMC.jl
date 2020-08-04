# NPSMC.jl

![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)<!--
![Lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-stable-green.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-retired-orange.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-archived-red.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-dormant-blue.svg) -->
[![Build Status](https://travis-ci.org/npsmc/NPSMC.jl.svg?branch=master)](https://travis-ci.org/npsmc/NPSMC.jl)
[![codecov](https://codecov.io/gh/npsmc/NPSMC.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/npsmc/NPSMC.jl)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://npsmc.github.io/NPSMC.jl/dev)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/npsmc/NPSMC.jl/master?filepath=https%3A%2F%2Fgithub.com%2Fnpsmc%2FNPSMC.jl%2Fnotebooks)

PLEASE NOTE THIS IS PRE-RELEASE SOFTWARE FOR PREVIEW PURPOSES ONLY

THIS SOFTWARE IS SUBJECT TO BREAKING CHANGES

**N**on **P**arametric **S**equential **M**onte-**C**arlo methods

This package is derived from the [¡AnDA!](https://github.com/ptandeo/anda) 
and [¡CEDA!](https://github.com/ptandeo/CEDA) Python libraries written by @ptandeo.  
These programs are attached to the following publications:

- (http://journals.ametsoc.org/doi/abs/10.1175/MWR-D-16-0441.1): Lguensat, R.,
Tandeo, P., Ailliot, P., Pulido, M., & Fablet, R. (2017). The Analog Data
Assimilation. *Monthly Weather Review*, 145(10), 4093-4107.

- (http://onlinelibrary.wiley.com/doi/10.1002/qj.3048/full): Dreano, D., Tandeo, P., Pulido, M., Ait‐El‐Fquih, B., Chonavel, T., & Hoteit, I. (2017). Estimating Model‐Error Covariances in Nonlinear State‐Space Models using Kalman Smoothing and the Expectation–Maximization Algorithm. *Quarterly Journal of the Royal Meteorological Society*, 143(705), 1877-1885.


## Installing NPSMC

```bash
git clone https://github.com/npsmc/NPSMC.jl
cd NPSMC.jl
julia --project
```

```julia
julia> using Pkg
julia> Pkg.instantiate()
julia> using IJulia
julia> notebook(dir=joinpath(pwd(),"notebooks"))
[ Info: running ...
```


## See also

 - [DataAssim.jl](https://github.com/Alexander-Barth/DataAssim.jl): Implementation of various ensemble Kalman Filter data assimilation methods in Julia.
 - [StateSpaceModels.jl](https://github.com/LAMPSPUC/StateSpaceModels.jl): Julia package for time-series analysis using state-space models.
- [EnKF.jl](https://github.com/mleprovost/EnKF.jl): Tools for data assimilation with Ensemble Kalman filter.
- [StateSpaceRoutines.jl](https://github.com/FRBNY-DSGE/StateSpaceRoutines.jl): Package implementing common state-space routines.
- [LowLevelParticleFilters.jl](https://github.com/baggepinnen/LowLevelParticleFilters.jl): Simple particle/kalman filtering, smoothing and parameter estimation.
- [ParticleFilters.jl](https://github.com/JuliaPOMDP/ParticleFilters.jl): Simple particle filter implementation in Julia
- [MiniKalman.jl](https://github.com/cstjean/MiniKalman.jl):  Kalman Filtering package
- [CovarianceMatrices.jl](https://github.com/gragusa/CovarianceMatrices.jl): Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation for Julia.
- [Francesco Martinuzzi posts]](https://martinuzzifrancesco.github.io/posts/)
