r"""
Region heuristic
================

Region heuristic is a method to propose a rectangular region in the complex 
plane that contains the most significant zeros of retarded or neutral 
quasi-polynomials. The heuristic is based on obtaining asymptotic exponentials
of the zero chains, bounds of the neutral vertical strip, and spectral envelope.
"""

import logging
import numpy as np
import numpy.typing as npt

from . import quasipoly
from .distribution.spectral_abscissa import _safe_upper_bound, _neutral_strip_bounds
from .distribution.envelope_curve import _envelope_imag_axis_crossing, _envelope_real_axis_crossing, _spectral_norms, _neutral_envelope_real_axis_crossing
from .distribution.zero_chains import chain_asymptotes

from .qpmr_validation import validate_qp

logger = logging.getLogger(__name__)

def region_heuristic(coefs, delays, **kwargs) -> tuple[float, float, float, float]:
    """ Proposes rectangular region that contains the most significant zeros
    of retarded or neutral quasi-polynomial.

    The heuristic is based on obtaining asymptotic exponentials of the zero 
    chains, bounds of the neutral vertical strip, and spectral envelope.

    Parameters
    ----------
    coefs : ndarray
        Matrix of polynomial coefficients. Each row represents the coefficients
        corresponding to a specific delay.

    delays : ndarray
        Vector of delays associated with each row in `coefs`.

    **kwargs : dict, optional
        n_roots : int, optional
            Approximate number of rightmost roots to contain in the proposed
            region. Default is 100.
        re_stretch : float, optional
            Stretch factor for the minimum real part of the region.
            Default is 0.05.

    Returns
    -------
    region : tuple of float
        Proposed region ``(Re_min, Re_max, Im_min, Im_max)``.

    Raises
    ------
    ValueError
        If the provided quasi-polynomial is advanced or if the heuristic cannot
        be applied to the given quasi-polynomial.

    Examples
    --------
    >>> import numpy as np
    >>> import qpmr
    >>> coefs = np.array([[1, 0], [0, 1]])
    >>> delays = np.array([0, 1])
    >>> region = qpmr.region_heuristic(coefs, delays, n_roots=50)
    >>> print(region)
    """
    # unpack kwargs
    n_roots = kwargs.get("n_roots", 100)
    re_stretch = kwargs.get("re_stretch", 0.05)

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
    im_max = 2 * np.pi / delays[-1] * n_roots

    if coefs.shape[1] == 1: # special case of quasi-polynomial associated with delay-difference equation
        # obtain reprezentation of normalized delay-difference equation
        # coefs[:,-1] = [1, a1, a2, ...], delays = [0, tau1, tau2, ...]
        # coefs are non-zero real numbers
        mask = coefs[:,-1] != 0
        diff_coefs, diff_delays = coefs[:,-1][mask], delays[mask]
        cdm, cdp = _neutral_strip_bounds(diff_coefs, diff_delays)

        region = (float((1 + re_stretch)*cdm), float((1 + re_stretch)*cdp),
                  0., float(im_max))
        return region

    # Next, treat quasi-polynomial as it is retarded
    # guess for re_max is obtained via envelope (connected to matrix norms of
    # RDDE) see the following book page 9:
    #   Michiels, Wim, and Silviu-Iulian Niculescu, eds. Stability, control,
    #   and computation for time-delay systems: an eigenvalue-based
    #   approach. Society for Industrial and Applied Mathematics, 2014.
    norms = _spectral_norms(coefs, delays)
    # we only care about point where envelope crosses real axis
    re_max = _envelope_real_axis_crossing(norms, delays)

    # re_min is obtained as the left-most crossing of im_max with all the
    # chain asymptotes
    mu, w = chain_asymptotes(coefs, delays)
    re_min = np.inf
    for i in range(len(mu)):
        for ww in w[i]:
            c = -mu[i] * np.log(im_max/ww)
            if c < re_min:
                re_min = c

    if np.any(coefs[1:, -1] != 0.): # quasi-polynomial is neutral
        # obtain reprezentation of normalized delay-difference equation
        # coefs[:,-1] = [1, a1, a2, ...], delays = [0, tau1, tau2, ...]
        # coefs are non-zero real numbers
        mask = coefs[:,-1] != 0
        diff_coefs, diff_delays = coefs[:,-1][mask], delays[mask]
        cdm, cdp = _neutral_strip_bounds(diff_coefs, diff_delays)

        # solve neutral envelope crossing with x0 = re_max obtained from retarded case
        re_max = _neutral_envelope_real_axis_crossing(norms[mask], diff_coefs, diff_delays, x0=re_max)
        re_max = max(re_max, cdp)
        re_min = min(re_min, cdm)

    # stretch Re min, by default 5% of the width of the region
    region = (float(re_min - re_stretch * abs(re_max - re_min)),
              float(re_max), 0., float(im_max))
    return region