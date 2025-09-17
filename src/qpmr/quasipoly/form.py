"""
Quasipolynomial forms
---------------------

"""

from enum import Enum
import logging
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


class QuasiPolynomialType(str, Enum):
    RETARDED = "RETARDED"
    NEUTRAL = "NEUTRAL"
    ADVANCED = "ADVANCED"

    POLYNOMIAL = "POLYNOMIAL"
    CONSTANT = "CONSTANT" # special case for h(s) = a this has no zeros

def _qp_type_from_normed(coefs: npt.NDArray, delays: npt.NDArray) -> QuasiPolynomialType:
    """ TODO """
    if delays.size == 0 or (delays.size == 1 and coefs.shape[1] <= 1):
        return QuasiPolynomialType.CONSTANT
    elif delays.size == 1:
        return QuasiPolynomialType.POLYNOMIAL
    elif coefs[0, -1] == 0.:
        return QuasiPolynomialType.ADVANCED
    elif np.any(coefs[1:, -1] != 0.):
        return QuasiPolynomialType.NEUTRAL
    else:
        return QuasiPolynomialType.RETARDED
    
def qp_type(coefs, delays):
    raise NotImplementedError
