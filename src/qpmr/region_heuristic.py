"""
Region heuristic
----------------

As of now, assuming retarded quasi-polynomial and goal is to find smallest
possible rectangular region that contains n rightmost roots.

"""

import logging
import numpy as np
import numpy.typing as npt

from .distribution.envelope_curve import _envelope_imag_axis_crossing, _envelope_real_axis_crossing

logger = logging.getLogger(__name__)

def region_heuristic(coefs, delays, n: int=50):

    # TODO make sure quasipolynomial is retarded and normed and compressed

    raise NotImplementedError(".")
    _envelope_real_axis_crossing()

    