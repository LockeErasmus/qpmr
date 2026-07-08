r"""
QPmR — Quasi-polynomial Root Finder
===================================

Finds roots of quasi-polynomials (delay differential characteristic equations)
in specified regions of the complex plane.

Public API
----------

- :func:`qpmr` — main root-finding algorithm (v3)
- :class:`QpmrInfo` — computation metadata returned by :func:`qpmr`
- :func:`region_heuristic` — propose a search region
- :func:`distribution_diagram`, :func:`chain_asymptotes` — spectrum distribution
- :func:`safe_upper_bound_diff`, :func:`bounds_neutral_strip` — spectral bounds
- :class:`QuasiPolynomial`, :class:`TransferFunction` — quasi-polynomial objects
- :func:`init_qpmr_logger` — configure package logging
- :mod:`qpmr.examples` — benchmark quasi-polynomials from the literature
"""

__version__ = "0.1.0"

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from .qpmr_v3 import qpmr, QpmrInfo
from .distribution import distribution_diagram, chain_asymptotes, safe_upper_bound_diff, bounds_neutral_strip

from .utils import init_qpmr_logger
from .utils import init_qpmr_logger as init_logger

from .quasipoly import QuasiPolynomial, TransferFunction
from .quasipoly import examples
