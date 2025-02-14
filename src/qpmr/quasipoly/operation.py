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
    ddelays = np.copy(delays)
    if m == 0:
        dcoefs = np.copy(coefs)
    if m == 1:
        dcoefs = (coefs.T * -delays).T
    else:
        dcoefs = (coefs.T * -delays).T
        dcoefs[:,:-1] = coefs[:,1:] * np.arange(1,m,1)
    
    if kwargs.get("compress", True):
        dcoefs, ddelays = compress(dcoefs, ddelays)
    
    return dcoefs, ddelays

def antiderivative(coefs: npt.NDArray, delays: npt.NDArray, **kwargs) -> tuple[npt.NDArray, npt.NDArray]:
    """ Antiderivative of quasipolynomial

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
    ddelays = np.copy(delays)
    if m == 0:
        raise NotImplementedError()
    if m == 1:
        raise NotImplementedError()
    else:
        raise NotImplementedError()
        dcoefs = (coefs.T * -delays).T
        dcoefs[:,:-1] = coefs[:,1:] * np.arange(1,m,1)

    if kwargs.get("compress", True):
        dcoefs, ddelays = compress(coefs, delays)
    
    return dcoefs, ddelays