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

from .qpmr_validation import validate_qp

logger = logging.getLogger(__name__)

def region_heuristic(coefs, delays, **kwargs) -> tuple[float, float, float, float]:
    """ TODO 
    
    Idea of this heuristic is that number of imaginary contours =approx= number
    of zeros in region. 
    


    """

    # unpack kwargs
    n_roots = kwargs("n_roots", 100)

    # validate quasi-polynomial
    coefs, delays = validate_qp(coefs, delays)

    # normalize quasi-polynomial
    coefs, delays = quasipoly.normalize(coefs, delays)

    # TODO - what if trivial, i.e. empty, constant or polynomial
    if coefs.size == 0 or len(delays) <= 1:
        raise NotImplementedError(f"Region heuristic cannot be used for polynomial (just use eigenvalue algorithm for companion matrix)")

    # filter out advanced type of quasi-polynomial - heuristic is not applicable
    # to those as they have exponential root chains going to the right 
    if coefs[0, -1] == 0:
        raise ValueError("Provided quasi-polynomial is advanced and region heuristic does not make sense (there is no bound for real part of spectrum)")
    
    # obtain the maximum imaginary bound via the idea that imaginary contours
    # =approx= number of zeros in the horizontal strip.  Next, we assume real
    # `coefs`, therefore spectra is symetrical by real axis -> Im = [0, im_max]
    im_min = 0
    im_max = 2 * np.pi / delays[-1] * n_roots

    # CASE 1: neutral spectrum  
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
    # CASE 2: retarded spectrum
    else: # retarded spectrum
        # re_max is obtained via envelope (connected to matrix norms of RDDE)
        # see book (TODO add page):
        #   Michiels, Wim, and Silviu-Iulian Niculescu, eds. Stability, control,
        #   and computation for time-delay systems: an eigenvalue-based
        #   approach. Society for Industrial and Applied Mathematics, 2014.
        norms = _spectral_norms(coefs, delays)
        # we only care about point where envelope crosses real axis
        re_max = _envelope_real_axis_crossing(norms, delays)

        # re_min is obtained as the left-most crossing of im_max with all the
        # chain asymptotes
        mu, w = chain_asymptotes(coefs, delays)
        re_min = -np.inf
        for i in range(len(mu)):
            for ww in w[i]:
                c = -mu[i] * np.log(im_max/ww)
                if c > re_min:
                    re_min = c
    # rectangle is stratched 5 % to the left, just small bit is enough so the
    # left most root chaim asymptote does not end precisely in top-left corner
    region = (re_min - 1.05 * abs(re_max - re_min), re_max, 0, im_max)
        
    return region