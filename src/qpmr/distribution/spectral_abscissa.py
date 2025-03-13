"""

"""
import logging

import numpy as np
from scipy.optimize import root_scalar # TODO, I do not want dependency of scipy!

from qpmr.quasipoly.core import compress, create_normalized_delay_difference_eq


logger = logging.getLogger(__name__)

def func(s, coefs, delays):
    """
    TODO
    """
    return np.inner(coefs, np.exp(-s*delays)) - 1.

def spectral_abscissa_diff(coefs, delays, **kwargs):
    """ TODO
    
    """
    if kwargs.get("compress", True):
        coefs, delays = compress(coefs, delays)
    
    diff = create_normalized_delay_difference_eq(coefs, delays, compress=False)
    logger.debug(f"{diff}")

    if diff is None:
        logger.warning("No associated delay difference equation -> cd= -INF")
        return -np.inf
    default_kwargs = {
        # "method": 'brentq',
        "x0": 0., # initial guess
    }
    sol = root_scalar(func, args=diff, **default_kwargs)
    return sol

    


