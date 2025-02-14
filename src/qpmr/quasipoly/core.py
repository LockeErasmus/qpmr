"""
TODO:
    1. is_empty QP
"""

import logging
from typing import Any

import numpy as np
import numpy.typing as npt


logger = logging.getLogger(__name__)

def _check_qp(coefs: npt.NDArray, delays: npt.NDArray, raise_error=True) -> bool:
    """ Checks if quasipolynomial definition is valid

    Args:
        coefs (array): matrix definition of polynomial coefficients (each row
            represents polynomial coefficients corresponding to delay)
        delays (array): vector definition of associated delays (each delay
            corresponds to row in `coefs`)
    """
    # TODO
    assert isinstance(coefs, np.ndarray)
    assert coefs.ndim == 2

    assert isinstance(delays, np.ndarray)
    assert coefs.ndim == 1

    assert coefs.shape[0] == delays.shape[0]

    return True
    

def _eval_scalar(coefs: npt.NDArray, delays: npt.NDArray, s: int|float|complex):
    """ Evaluates quasipolynomial on complex value

    Args:
        coefs (array): matrix definition of polynomial coefficients (each row
            represents polynomial coefficients corresponding to delay)
        delays (array): vector definition of associated delays (each delay
            corresponds to row in `coefs`)
        s (number): complex value to evaluate quasipolynomial

    Returns:
        number: evaluated quasipolynomial at `s`
    """
    powers = np.arange(0, coefs.shape[1], 1, dtype=int)
    return np.inner(np.sum(coefs * np.power(s,  powers), axis=1), np.exp(-delays*s))

def _eval_array(coefs: npt.NDArray, delays: npt.NDArray, s: npt.NDArray):
    """ Evaluates quasipolynomial on nD array of complex values

    Args:
        coefs (array): matrix definition of polynomial coefficients (each row
            represents polynomial coefficients corresponding to delay)
        delays (array): vector definition of associated delays (each delay
            corresponds to row in `coefs`)
        s (array): nD array of complex values to evaluate quasipolynomial on

    Returns:
        array: evaluated quasipolynomial at all elements of `s`
    """
    coefs = coefs.T # transpose rows - powers of s, cols - delays
    delays = delays
    powers = np.arange(0, coefs.shape[0], 1, dtype=int)
    dels = np.exp(- s[..., np.newaxis] * delays[np.newaxis, ...])
    aa = dels[..., np.newaxis] * coefs.T[np.newaxis, ...] # (..., n_delays, order)
    r = np.multiply(
        np.power(s[..., np.newaxis], powers[np.newaxis, ...]), # (..., order)
        np.sum(aa, axis=-2), # sum by n_delays axis -> (..., order)
    )
    return np.sum(r, axis=-1)

def eval(coefs: npt.NDArray, delays: npt.NDArray, s: Any):
    """ Evaluates quasipolynomial on s

    Args:
        coefs (array): matrix definition of polynomial coefficients (each row
            represents polynomial coefficients corresponding to delay)
        delays (array): vector definition of associated delays (each delay
            corresponds to row in `coefs`)
        s (array or number): complex value (s) to evaluate quasipolynomial on

    Returns:
        array: evaluated quasipolynomial at all elements of `s`
    """
    if isinstance(s, (int, float, complex)):
        return _eval_scalar(coefs, delays, s)
    elif isinstance(s, np.ndarray):
        return _eval_array(coefs, delays, s)
    else:
        raise ValueError(f"Unsupported type of s '{type(s)}'")

def compress(coefs: npt.NDArray, delays: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
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

def poly_degree(poly: npt.NDArray, order="reversed") -> int:
    """ assumes 1D array as input
    
    [a0, a1, ... am, 0, 0, ... 0] -> degree m

    reverse order

    """
    degree = len(poly) - 1
    if order == "reversed":
        poly_ = poly[::-1]
    else:
        poly_ = poly
    for a in poly_:
        if a != 0.:
            break
        degree -= 1
    logger.debug(f"{poly=} -> degree: {degree}")
    return degree
