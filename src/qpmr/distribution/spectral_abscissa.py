"""

"""
import logging

import numpy as np

from qpmr.quasipoly.core import compress, create_normalized_delay_difference_eq
from qpmr.numerical_methods import newton

logger = logging.getLogger(__name__)

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
        Df=lambda s: np.inner(-diff_delays*diff_coefs, np.exp(-s*diff_delays)),
        x0=0.,
        epsilon=1e-6,
        max_iter=100,
    )

    return bound

    




