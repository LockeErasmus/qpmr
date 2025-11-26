"""
Region heuristic
----------------

As of now, assuming retarded quasi-polynomial and goal is to find smallest
possible rectangular region that contains n rightmost roots.

"""

import logging
import numpy as np
import numpy.typing as npt

from . import quasipoly
from .distribution.spectral_abscissa import _safe_upper_bound
from .distribution.envelope_curve import _envelope_imag_axis_crossing, _envelope_real_axis_crossing, _spectral_norms
from .distribution.zero_chains import chain_asymptotes

logger = logging.getLogger(__name__)

def region_heuristic(coefs, delays, n: int=50) -> tuple[float, float, float, float]:
    """ TODO """

    # step one, normalize quasi-polynomial
    coefs, delays = quasipoly.normalize(coefs, delays)

    # TODO - what if trivial, i.e. empty, constant or polynomial
    if coefs.size == 0 or len(delays) <= 1:
        raise NotImplementedError(f"Not a quasipolynomial")

    # solve type of quasi-polynomial
    if coefs[0, -1] == 0:
        raise ValueError("Provided quasi-polynomial is advanced and region heuristic does not make sense (there is no bound for real part of spectrum)")
    
    # extract delay-difference associated exponential
    # Notes:
    #   (1) the equation is already normalized
    #   (2) coefs at least two rows (one row -> finite spectrum = polynomial)
    #
    
    if np.any(coefs[1:, -1] != 0.): # neutral spectrum
        # first, obtain exponential sum associated with normalized delay-difference equation
        # coefs[:,-1] = [1, a1, a2, ...], delays = [0, tau1, tau2, ...]
        mask = coefs[:,-1] != 0
        mask[0] = False # first coefficient is 1 -> fix theta to solve smaller problems
        ndiff_coefs, ndiff_delays = coefs[:,-1][mask], delays[mask]
        bound_right = _safe_upper_bound(ndiff_coefs, ndiff_delays, x0=0, tol=1e-6, max_iter=100)
        bound_left = _safe_upper_bound(ndiff_coefs, -ndiff_delays, x0=0, tol=1e-6, max_iter=100)
        logger.debug(f"Quasi-polynomial is of NEUTRAL type, neutral spectrum is bounded inside [{bound_left}, {bound_right}]")

        # TODO, solve envelope curve

        raise NotImplementedError(f"Neutral not implemented yet")

    else: # retarded spectrum
        # envelope to obtain Re_max
        # mi_vec, abs_omega = chain_asymptotes(coefs, delays)
        norms = _spectral_norms(coefs, delays)
        re_max = _envelope_real_axis_crossing(norms, delays)

        # calculation
        tau_max = delays[-1]
        im_max = 2 * np.pi/tau_max * n # Tomas wants default to 50, but I think it should be more like 50 from each 

        # obtain re_min
        mu, w = chain_asymptotes(coefs, delays)

        points = []
        for i in range(len(mu)):
            for ww in w[i]:
                # calculate intersection with im_max
                # TODO - we can estimate well using 
                points.append(
                    -mu[i] * np.log(im_max/ww)
                )

        re_min = min(points) - 2*np.pi/tau_max
        
        region = (re_min, re_max, 0, im_max)
        return region