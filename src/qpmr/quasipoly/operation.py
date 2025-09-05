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

def derivative(coefs: npt.NDArray, delays: npt.NDArray, **kwargs) -> tuple[npt.NDArray, npt.NDArray]:
    """ Derivative of quasipolynomial

    Assume quasipolynomial h(s) in a form of sum of products of polynomials of
    order m_i <= m and exp(.)
    
                n
        h(s) = SUM p_i(s) * exp(-tau_i*s)
               i=0

    derivative can be solved via product rule applied on all those products.

    Args:
        coefs (array): matrix definition of polynomial coefficients (each row
            represents polynomial coefficients corresponding to delay)
        delays (array): vector definition of associated delays (each delay
            corresponds to row in `coefs`)
        **kwargs:
            compress (bool): if True compresses the result (converts to minimal
                form), default True
    
    Returns:
        tuple containing:

            - coefs (array): matrix definition of polynomial coefficients (each row
                represents polynomial coefficients corresponding to delay)
            - delays (array): vector definition of associated delays (each delay
                corresponds to row in `coefs`)
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




def _solve_upper_triangular(A, b):
    """
    Solves Ax = b for x, where A is an upper triangular matrix.
    No external dependencies.
    """
    n = len(b)
    x = np.zeros_like(b, dtype=float)

    for i in reversed(range(n)):
        
        if A[i, i] == 0:
            raise ValueError(f"Zero on diagonal at index {i}; system is singular.")
        
        
        sum_ax = np.dot(A[i, i+1:], x[i+1:])
        x[i] = (b[i] - sum_ax) / A[i, i]
    return x

def antiderivative(coefs: npt.NDArray, delays: npt.NDArray, **kwargs) -> tuple[npt.NDArray, npt.NDArray]:
    """ Antiderivative of quasipolynomial

    Assume quasipolynomial h(s) in a form of sum of products of polynomials of
    order m_i <= m and exp(.)
    
                n
        h(s) = SUM p_i(s) * exp(-tau_i*s)
               i=0

    antiderivative can be solved ... TODO

    Args:
        coefs (array): matrix definition of polynomial coefficients (each row
            represents polynomial coefficients corresponding to delay)
        delays (array): vector definition of associated delays (each delay
            corresponds to row in `coefs`)
        **kwargs:
            compress (bool): if True compresses the result (converts to minimal
                form), default True
    
    Returns:
        tuple containing:

            - coefs (array): matrix definition of polynomial coefficients (each row
                represents polynomial coefficients corresponding to delay)
            - delays (array): vector definition of associated delays (each delay
                corresponds to row in `coefs`)
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