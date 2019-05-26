# NPSMC.jl

[![Build Status](https://travis-ci.org/npsmc/NPSMC.jl.svg?branch=master)](https://travis-ci.org/npsmc/NPSMC.jl)
[![codecov](https://codecov.io/gh/npsmc/NPSMC.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/npsmc/NPSMC.jl)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://npsmc.github.io/NPSMC.jl/dev)

Non parametric sequential Monte-Carlo methods

## Installing NPSMC

In a Julia session switch to `pkg>` mode to add `NPSMC`:

```julia
julia>] # switch to pkg> mode
pkg> add https://github.com/npsmc/NPSMC.jl
```

Alternatively, you can achieve the above using the `Pkg` API:

```julia
julia> using Pkg
julia> pkg"add https://github.com/npsmc/NPSMC.jl"
```

When finished, make sure that you're back to the Julian prompt (`julia>`)
and bring `NPSMC` into scope:

```julia
julia> using NPSMC
```
