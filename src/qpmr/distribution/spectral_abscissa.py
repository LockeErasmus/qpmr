"""

"""
import logging

import numpy as np

from qpmr.quasipoly.core import compress, create_normalized_delay_difference_eq
from qpmr.core.quasipolynomial import compress, extract_delay_diff_eq
from qpmr.numerical_methods import newton

logger = logging.getLogger(__name__)


def _safe_upper_bound(ndiff_coefs, ndiff_delays, x0, tol, max_iter):
    coefs_abs = np.abs(ndiff_coefs)
    bound, converged = newton(
        f=lambda x: np.inner(coefs_abs, np.exp(-x*ndiff_delays)) - 1.,
        f_prime=lambda x: np.inner(-ndiff_delays*coefs_abs, np.exp(-x*ndiff_delays)),
        x0=x0,
        tol=tol,
        max_iter=max_iter,
    )
    return bound

def _neutral_strip_bounds(diff_coefs, diff_delays, **kwargs) -> tuple[float, float]:
    """
    Returns bounds for vertical strip associated with neutral segment
    """
    ub = _safe_upper_bound(diff_coefs[1:]/diff_coefs[0], diff_delays[1:] - diff_delays[0], 0, 1e-6, 100)
    lb = -_safe_upper_bound(diff_coefs[:-1]/diff_coefs[-1], -diff_delays[:-1] + diff_delays[-1], 0, 1e-6, 100)
    return lb, ub

# def _neutral_strip_bounds(ndiff_coefs, ndiff_delays, **kwargs) -> tuple[float, float]:
#     """
#     Docstring for delay_difference_eq_bounds

#     :param ndiff_coefs: Description
#     :param ndiff_delays: Description
#     :param kwargs: Description

#     assumes coefs, delays define normalized delay-difference equation
#     returns bounds for vertical strip associated with neutral segment

#     """
#     ub = _safe_upper_bound(ndiff_coefs, ndiff_delays, 0, 1e-6, 100)
#     lb = -_safe_upper_bound(ndiff_coefs[:-1]/ndiff_coefs[-1], -ndiff_delays[:-1] + ndiff_delays[-1], 0, 1e-6, 100)
#     return lb, ub

def safe_upper_bound_diff(coefs, delays, **kwargs):
    """ TODO
    
    """
    if kwargs.get("compress", True):
        coefs, delays = compress(coefs, delays)
    
    diff = create_normalized_delay_difference_eq(coefs, delays, compress=False)
    logger.debug(f"{diff}")
    if diff is None:
        return -np.inf
    
    diff_coefs, diff_delays = diff # unpack
    
    # bound, f, counter = chandrupatla(
    #     lambda s: np.inner(diff_coefs, np.exp(-s*diff_delays)) - 1.,
    #     x1=-100,
    #     x2=100.,
    # )
    bound, converged = newton(
        f=lambda s: np.inner(diff_coefs, np.exp(-s*diff_delays)) - 1.,
        f_prime=lambda s: np.inner(-diff_delays*diff_coefs, np.exp(-s*diff_delays)),
        x0=0.,
        tol=1e-6,
        max_iter=100,
    )

    return bound

def bounds_neutral_strip(coefs, delays, **kwargs):
    """ TODO
    
    """
    if kwargs.get("compress", True):
        coefs, delays = compress(coefs, delays)

    diff_coefs, diff_delays = extract_delay_diff_eq(coefs, delays, normalize=True, compress=False)
    if len(diff_coefs) <= 1:
        return np.inf, -np.inf # no neutral strip, inf, -inf as this is consistent with the definition
    
    ub, lb = _neutral_strip_bounds(diff_coefs, diff_delays)
    return ub, lb
    




