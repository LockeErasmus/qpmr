# Roadmap

## Concrete fixes

1. ~~truncating roots to certain precision ???~~


## Features

1. Automatic `region` selection, by default, it should capture `50` rightmost roots
    - `im_min=0` by default
    - obtain `re_max` from envelope
    - point (`re_min`, `im_max`) obtained from guessing number of roots via asymptote of the root chains, core idea:
        - for each asymptote, find point where the approximating asymptote is "close" (we will use this, because after we hit "close" we will assume certain distribution of roots on the asymptote)
        - before this point is hit, maybe argument principle?
    - if neutral, use safe upper bound also for right point
1. Allow for **not-storing** partial results in no-leaf nodes
1. Evaluate argument principle at the begining to disregard subregions
1. Custom Errors (for instance newton did not converge) and their handling, to allow return results eventhough one leaf fails



## Task to implement
 - quasipolynomial arithmetic operations
 - `QuasiPoly` class with convenient interface and everything
 - `TransferFunction` class with convenient interface and everything
 - tests
 - examples (notebooks and py files)
 - example polynomials and quasipolynomials

## Tasks to consider implement

- `qpmr/numba`
- symbolical version of `qpmr`, `qpmr/symbolic` ?
- root multiplicity
- also version for fractional order
- GPU acceleration
- add aQPmR functionality to find regions free of roots
- QPmR based root-locus
- stability envelope -> induced matrix norm

## Questions for Tomas



## Simillar python libraries

1. `cxtroots`
1. julia interval arithmetics