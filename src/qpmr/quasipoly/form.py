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


def _sort():
    raise NotImplementedError

def _compress():
    raise NotImplementedError

def _shift_delays():
    raise NotImplementedError

def _normalize_to_monic():
    raise NotImplementedError

def _qp_type_from_arrays(coefs: npt.NDArray, delays: npt.NDArray, **kwargs) -> QuasiPolynomialType:
    raise NotImplementedError



def _compress2(coefs: npt.NDArray, delays: npt.NDArray, **kwargs):
    """

    """
    """ Compresses quasipolynomial representation into a form where no
    duplicates in delays are present and last column vector of `coefs` is
    non-zero. Compressed quasipolynomial has also ordered delays in ascending
    order.

    Args:
        coefs (array): matrix definition of polynomial coefficients (each row
            represents polynomial coefficients corresponding to delay)
        delays (array): vector definition of associated delays (each delay
            corresponds to row in `coefs`)
    
    Returns:
        tuple containing:

            - coefs (array): matrix definition of polynomial coefficients (each row
                represents polynomial coefficients corresponding to delay)
            - delays (array): vector definition of associated delays (each delay
                corresponds to row in `coefs`)

    """
    logger.debug(f"Original quasipolynomial:\n{coefs}\n{delays}")
    if coefs.size == 0 and delays.size == 0: # trivial case with empty definition, i.e. qp(s) = 0
        # note that new arrays are constructed, because if coefs are of shape
        # for instance (N,0) I want always the shape to be (1,0)
        return np.array([[]], dtype=coefs.dtype), np.array([], dtype=delays.dtype)
    
    delays_compressed = np.unique(delays) # sorted 1D array of unique delays
    n, m = delays_compressed.shape[0], coefs.shape[1]
    coefs_compressed = np.zeros(shape=(n, m), dtype=coefs.dtype)
    
    for i in range(n):
        mask = (delays == delays_compressed[i])
        coefs_compressed[i, :] = np.sum(coefs[mask, :], axis=0, keepdims=False)
    
    # at this point, representation is unique in delays, we need to make sure
    # `coefs_compressed` does not have: 1) row full of zeros and 2) last column
    #  full of zeros
    col_mask = ~(coefs_compressed == 0).all(axis=0) # True if column has at least one non-zero
    ix = np.argmax(col_mask[::-1]) # first occurence of True indexed from end
    col_mask = np.full_like(col_mask, fill_value=True, dtype=bool)
    if ix > 0: # at least one column from back should be deleted
        col_mask[-ix:] = False
    row_mask = ~(coefs_compressed == 0).all(axis=1) # True if row has atleast one non-zero coefficient

    coefs_compressed = coefs_compressed[np.ix_(row_mask, col_mask)]
    delays_compressed = delays_compressed[row_mask]
    logger.debug((f"Compressed quasipolynomial\n{coefs_compressed}"
                  f"\n{delays_compressed}"))
    if not coefs_compressed.size: # resulting qp is empty
        return np.array([[]], dtype=coefs.dtype), np.array([], dtype=delays.dtype)

    return coefs_compressed, delays_compressed