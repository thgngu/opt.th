# cvx-optim.torch • [ ![Build Status] [travis-image] ] [travis] [ ![License] [license-image] ] [license]

*Torch library for convex optimization.*

[travis-image]: https://travis-ci.org/bamos/cvx-optim.torch.png?branch=master
[travis]: http://travis-ci.org/bamos/cvx-optim

[license-image]: http://img.shields.io/badge/license-Apache--2-blue.svg?style=flat
[license]: LICENSE

---

> Thanks for your interest in this project!
> It's a work-in-progress and I have a lot more planned in this space,
> including more algorithms, mini-batch support, and Python implementations.
> I'll post on [my blog](http://bamos.github.io/blog/)
> ([RSS](http://bamos.github.io/atom.xml)) with some
> visualizations and a release when I'm finished.
>
> -Brandon. [2016-07-20]

---

You may also be interested in:
+ Torch [Gurobi](http://www.gurobi.com/) bindings at
  [bamos/gurobi.torch](https://github.com/bamos/gurobi.torch).
+ Torch [ECOS](https://github.com/embotech/ecos) bindings at
  [bamos/ecos.torch](https://github.com/bamos/ecos.torch).

---

# Algorithms

## Spectral Projected Gradient (SPG)

`cvx-optim.spg` implements the Spectral Projected Gradient (SPG)
method [1,2] for convex-constrained optimization.

## Projected Gradient Descent (PGD)

`cvx-optim.pgd` implements projected gradient descent (PGD) for
convex-constrained optimization with the proximal algorithm
described in Section 4.3 of [3].

## Bundle Method (in progress)

Algorithm 1 of [4].

# Installation

After installing Torch, `cvx-optim.torch` can be installed with:

1. `luarocks install https://raw.githubusercontent.com/bamos/cvx-optim.torch/master/cvx-optim-scm-1.rockspec`
2. Cloning this repo and running `luarocks make`.

# Tests

After installing the library with `luarocks`, our tests in
[test.lua](https://github.com/bamos/cvx-optim.torch/blob/master/test.lua)
can be run with `th test.lua`.

# Licensing

This repository is
[Apache-licensed](https://github.com/bamos/cvx-optim.torch/blob/master/LICENSE).

# References

```
[1] Birgin, Ernesto G., José Mario Martínez, and Marcos Raydan.
"Algorithm 813: SPG—software for convex-constrained optimization."
ACM Transactions on Mathematical Software (TOMS) 27.3 (2001): 340-349.
http://www.ime.usp.br/~egbirgin/publications/bmr2.pdf

[2] Birgin, Ernesto G., José Mario Martínez, and Marcos Raydan.
"Nonmonotone spectral projected gradient methods on convex sets."
SIAM Journal on Optimization 10.4 (2000): 1196-1211.
http://epubs.siam.org/doi/pdf/10.1137/S1052623497330963

[3] Parikh, Neal, and Stephen P. Boyd.
"Proximal Algorithms."
Foundations and Trends in optimization 1.3 (2014): 127-239.
http://www.web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf

[4] Smola, Alex J., S.V.N. Vishwanathan, and Quoc V. Le
"Bundle methods for machine learning."
Advances in Neural Information Processing Systems. 2007.

http://machinelearning.wustl.edu/mlpapers/paper_files/NIPS2007_470.pdf
```
