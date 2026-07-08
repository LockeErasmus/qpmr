"""
More complex operations on quasipolynomials
-------------------------------------------

TODO:
    1. antiderivative
"""

import logging

import numpy as np
import numpy.typing as npt

from .core import compress

logger = logging.getLogger(__name__)


def _retarded_qp2ss(coefs: npt.NDArray, delays: npt.NDArray):
    """Convert a retarded quasi-polynomial to companion state-space form."""
    # TODO assert that arrays are normalized and compressed

    n, d = coefs.shape
    A = np.zeros(shape=(n-1, n-1, d), dtype=coefs.dtype)
    
    A[1:, :-1, 0] = np.eye(n-2)
    A[-1, :, :] = -coefs[:, :-1].T

    return A, delays


# def _shift_matrix(a: float, order: int):
#     """ """
#     if order == 0:
#         raise NotImplementedError
    

# def shift(coefs, delays, **kwargs)-> tuple[npt.NDArray, npt.NDArray]:
#     """ shifts quasipolynomial in real direction 
    
    
#     Notes
#     -----

#     Assuming input is quasi-polynomial :math:h(s) then the result is 
    
#     """


def derivative(coefs: npt.NDArray, delays: npt.NDArray, **kwargs) -> tuple[npt.NDArray, npt.NDArray]:
    """Differentiate a quasi-polynomial with respect to ``s``.

    Applies the product rule to each term
    :math:`p_i(s) e^{-\\tau_i s}`.

    Parameters
    ----------
    coefs : ndarray
        Matrix of polynomial coefficients. Each row represents the coefficients
        corresponding to a specific delay.
    delays : ndarray
        Vector of delays associated with each row in ``coefs``.
    compress : bool, optional
        If ``True`` (default), compress the result.

    Returns
    -------
    coefs_prime : ndarray
        Derivative coefficient matrix.
    delays_prime : ndarray
        Derivative delay vector (unchanged from input).
    """
    n, m = coefs.shape
    coefs_prime = np.zeros_like(coefs)
    delays_prime = np.copy(delays)

    # iterate over column
    # first column is special
    coefs_prime[:, 0] -= delays * coefs[:, 0]
    # columns 1 ... m
    for j in range(1, m):
        coefs_prime[:, j-1] += j * coefs[:, j]
        coefs_prime[:, j] -= delays * coefs[:, j]
    
    if kwargs.get("compress", True):
        coefs_prime, delays_prime = compress(coefs_prime, delays_prime)
    
    return coefs_prime, delays_prime

def antiderivative(coefs: npt.NDArray, delays: npt.NDArray, **kwargs) -> tuple[npt.NDArray, npt.NDArray]:
    """Integrate a quasi-polynomial with respect to ``s``.

    Parameters
    ----------
    coefs : ndarray
        Matrix of polynomial coefficients. Each row represents the coefficients
        corresponding to a specific delay.
    delays : ndarray
        Vector of delays associated with each row in ``coefs``.
    compress : bool, optional
        If ``True`` (default), compress the result.

    Returns
    -------
    coefs_anti : ndarray
        Antiderivative coefficient matrix (one extra column).
    delays_anti : ndarray
        Antiderivative delay vector (unchanged from input).
    """
    # TODO I am assuming at least one coefficient one delay - if else for empty quasi-polynomial

    n, m = coefs.shape
    coefs_anti = np.zeros(shape=(n, m+1))
    delays_anti = np.copy(delays)
    zero_delay_mask = delays == 0.0
    nonzero_delay_mask = delays != 0.0

    # solve antiderivative of polynomials (tau = 0.0)
    np.divide(coefs, np.arange(1, m+1, 1, dtype=float), out=coefs_anti[:, 1:], where=zero_delay_mask[:, None])
    # solve antiderivative of quasipolynomials (tau =/= 0)
    # this is equivalent to sovling Ax=b where A is upper bidiagonal matrix and
    # can be done via simple elimination (no need for inverse)
    np.divide(coefs[:, -1], -delays, out=coefs_anti[:, m-1], where=nonzero_delay_mask)
    for j in range(m-2, -1, -1):
        np.divide(coefs[:, j] - (j+1)*coefs_anti[:, j+1],
                  -delays, out=coefs_anti[:, j], where=nonzero_delay_mask)    

    if kwargs.get("compress", True):
        coefs_anti, delays_anti = compress(coefs_anti, delays_anti)
    
    return coefs_anti, delays_anti