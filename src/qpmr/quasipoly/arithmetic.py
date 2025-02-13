"""
Basic arithmetic operations on quasipolynomials
-----------------------------------------------
add
multiplication
"""

import logging

import numpy as np
import numpy.typing as npt

from .core import compress

logger = logging.getLogger(__name__)

def add(coefs1: npt.NDArray, delays1: npt.NDArray, coefs2: npt.NDArray, delays2: npt.NDArray, **kwargs) -> tuple[npt.NDArray, npt.NDArray]:
    """ Adds two quasipolynomials without checks

    Args:
        coefs1 (array): matrix definition of polynomial coefficients (each row
            represents polynomial coefficients corresponding to delay)
        delays1 (array): vector definition of associated delays (each delay
            corresponds to row in `coefs`)
        coefs2 (array): matrix definition of polynomial coefficients (each row
            represents polynomial coefficients corresponding to delay)
        delays2 (array): vector definition of associated delays (each delay
            corresponds to row in `coefs`)

    Returns:
        tuple containing:

            - coefs (array): matrix definition of polynomial coefficients (each row
                represents polynomial coefficients corresponding to delay)
            - delays (array): vector definition of associated delays (each delay
                corresponds to row in `coefs`)
    """
    n1, m1 = coefs1.shape # ndelays x degree*
    n2, m2 = coefs2.shape

    if m1 >= m2:
        coefs = np.zeros(shape=(n1+n2, m1), dtype=coefs1.dtype)
        coefs[:n1, :] = coefs1
        coefs[n1:, :m2] = coefs2
        delays = np.r_[delays1, delays2]
        if kwargs.get("compress", True):
            coefs, delays = compress(coefs, delays)
        return coefs, delays
    else:
        return add(coefs2, delays2, coefs1, delays1)

def multiply(coefs1: npt.NDArray, delays1: npt.NDArray, coefs2: npt.NDArray, delays2: npt.NDArray, **kwargs) -> tuple[npt.NDArray, npt.NDArray]:
    """ Multiply two quasipolynomials without checks
    
    Args:
        coefs1 (array): matrix definition of polynomial coefficients (each row
            represents polynomial coefficients corresponding to delay)
        delays1 (array): vector definition of associated delays (each delay
            corresponds to row in `coefs`)
        coefs2 (array): matrix definition of polynomial coefficients (each row
            represents polynomial coefficients corresponding to delay)
        delays2 (array): vector definition of associated delays (each delay
            corresponds to row in `coefs`)

    Returns:
        tuple containing:

            - coefs (array): matrix definition of polynomial coefficients (each row
                represents polynomial coefficients corresponding to delay)
            - delays (array): vector definition of associated delays (each delay
                corresponds to row in `coefs`)
    """
    n1, m1 = coefs1.shape # ndelays x degree*
    n2, m2 = coefs2.shape

    # TODO empty ?
    # if one is zero -> zero

    coefs = np.zeros(shape=(n1*n2, m1+m2-1), dtype=coefs1.dtype)
    delays = np.zeros(shape=(n1*n2,), dtype=delays1.dtype)

    for i in range(n1):
        for j in range(n2):
            row_ix = i*n2 + j
            coefs[row_ix, :] = np.convolve(coefs1[i,:], coefs2[j,:], mode="full")
            delays[row_ix] = delays1[i] + delays2[j]
    
    if kwargs.get("compress", True):
        coefs, delays = compress(coefs, delays)
    return coefs, delays

