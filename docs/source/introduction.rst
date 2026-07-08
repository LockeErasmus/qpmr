Introduction
============

**qpmr** is a Python package for finding roots of quasi-polynomials
(characteristic equations of time-delay systems) in a region of the complex
plane. It implements an improved version of the mapping-based QPmR algorithm of
:cite:`vyhlidal2009mapping`, together with tools for spectrum distribution
analysis, quasi-polynomial algebra, and plotting.

Quasi-polynomials are given by coefficient matrices ``coefs`` and delay vectors
``delays``. The main entry point is :func:`qpmr.qpmr`.

.. bibliography::
