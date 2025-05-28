# Roadmap

## Concrete fixes

- truncating roots to certain precision ???

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

- argument principle and zeros located precisely on boundaries
- what is the reason for ds <- 1/3*ds rule
- where is the proof for grid heuristic?

## Simillar python libraries

1. `cxtroots`
1. julia interval arithmetics