"""
Basic arithmetic operations on quasipolynomials
-----------------------------------------------
As of now:
    1. addition
    2. multiplication

Notes:
    1. negative quasipolynomial can be constructed simply by `-coefs`
    2. subtraction = addition + negative
    3. division of two quasipolynomials is meromophic function -> not covered
"""

import logging

import numpy as np
import numpy.typing as npt

from .core import compress

logger = logging.getLogger(__name__)

def add(coefs1: npt.NDArray, delays1: npt.NDArray, coefs2: npt.NDArray, delays2: npt.NDArray, **kwargs) -> tuple[npt.NDArray, npt.NDArray]:
    """Add two quasi-polynomials.

    Parameters
    ----------
    coefs1, coefs2 : ndarray
        Matrix of polynomial coefficients. Each row represents the coefficients
        corresponding to a specific delay.
    delays1, delays2 : ndarray
        Vector of delays associated with each row in the corresponding
        coefficient matrix.
    compress : bool, optional
        If ``True`` (default), compress the result.

    Returns
    -------
    coefs : ndarray
        Sum coefficient matrix.
    delays : ndarray
        Sum delay vector.
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
    """Multiply two quasi-polynomials.

    Parameters
    ----------
    coefs1, coefs2 : ndarray
        Matrix of polynomial coefficients. Each row represents the coefficients
        corresponding to a specific delay.
    delays1, delays2 : ndarray
        Vector of delays associated with each row in the corresponding
        coefficient matrix.
    compress : bool, optional
        If ``True`` (default), compress the result.

    Returns
    -------
    coefs : ndarray
        Product coefficient matrix.
    delays : ndarray
        Product delay vector.
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

