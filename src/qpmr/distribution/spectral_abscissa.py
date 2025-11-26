"""

"""
import logging

import numpy as np

from qpmr.quasipoly.core import compress, create_normalized_delay_difference_eq
from qpmr.numerical_methods import newton

logger = logging.getLogger(__name__)



def _safe_upper_bound(coefs, delays, x0, tol, max_iter):
    coefs_abs = np.abs(coefs)
    bound, converged = newton(
        f=lambda x: np.inner(coefs_abs, np.exp(-x*delays)) - 1.,
        f_prime=lambda x: np.inner(-delays*coefs_abs, np.exp(-x*delays)),
        x0=x0,
        tol=tol,
        max_iter=max_iter,
    )
    return bound


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

    




