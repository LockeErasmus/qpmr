"""
Region heuristic
----------------

As of now, assuming retarded quasi-polynomial and goal is to find smallest
possible rectangular region that contains n rightmost roots.

"""

import logging
import numpy as np
import numpy.typing as npt


from .argument_principle import argument_principle_rectangle

from .distribution.envelope_curve import _envelope_imag_axis_crossing, _envelope_real_axis_crossing
from .distribution.psi_curve import psi_comensurate_0_kappa

logger = logging.getLogger(__name__)

def region_heuristic(coefs, delays, n: int=50):

    
    logger.debug(f"")

    # determine safe_upper_bound
    # determine envelope
    # 




    # # TODO make sure quasipolynomial is retarded and normed and compressed
    
    # base_delay = 0.1
    # grid_points = 5
    # jhh = np.pi / (grid_points * n_k[-1])

    # n_k = np.round(delays / base_delay, decimals=0).astype(int)

    # A, delays = _retarded_qp2ss(coefs, delays)

    # for shift in range(5):

    #     for k in range(grid_points*n_k[-1]):

    #         r, _ = np.linalg.eig(

    #         )

    #         A * np.exp(-delays*)
