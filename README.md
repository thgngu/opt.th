# spg.torch • [ ![Build Status] [travis-image] ] [travis] [ ![License] [license-image] ] [license]

*Torch Spectral Projected Gradient (SPG) Methods for Convex-Constrained Optimization*

[travis-image]: https://travis-ci.org/bamos/spg.torch.png?branch=master
[travis]: http://travis-ci.org/bamos/spg.torch

[license-image]: http://img.shields.io/badge/license-Apache--2-blue.svg?style=flat
[license]: LICENSE

# Spectral Projected Gradient

This is a Torch implementation of SPG [1,2] for
convex-constrained optimization that solves problems of the form

```
min f(x) subject to x∈Ω
```

where Ω is a closed convex set and
f is defined and has continuous partial derivatives
on an open set than contains Ω.

```
[1] Birgin, Ernesto G., José Mario Martínez, and Marcos Raydan.
"Algorithm 813: SPG—software for convex-constrained optimization."
ACM Transactions on Mathematical Software (TOMS) 27.3 (2001): 340-349.
http://www.ime.usp.br/~egbirgin/publications/bmr2.pdf

[2] Birgin, Ernesto G., José Mario Martínez, and Marcos Raydan.
"Nonmonotone spectral projected gradient methods on convex sets."
SIAM Journal on Optimization 10.4 (2000): 1196-1211.
http://epubs.siam.org/doi/pdf/10.1137/S1052623497330963

# Installation

After installing Torch, `spg.torch` can be installed with:

1. `luarocks install https://raw.githubusercontent.com/bamos/spg.torch/master/spg-scm-1.rockspec`
2. Cloning this repo and running `luarocks make`.

# Usage

The following solves the linear program

# Tests

After installing the library with `luarocks`, our tests in
[test.lua](https://github.com/bamos/spg.torch/blob/master/test.lua)
can be run with `th test.lua`.

# Licensing

This repository is
[Apache-licensed](https://github.com/bamos/gurobi.torch/blob/master/LICENSE).
