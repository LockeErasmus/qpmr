"""
Newton's method
---------------
Notes:
    1. vectorized version easily obtainable
"""
import logging
from typing import Callable

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

def _mueller_scalar(f: Callable, f_prime: Callable, x0: float | complex, m: float=1.0, tol: float=1e-8, max_iter: int=100):
    """Scalar Müller iteration (not yet implemented)."""
    raise NotImplementedError

def _mueller_array(f: Callable, x0: npt.NDArray, x1: npt.NDArray, x2: npt.NDArray, tol: float=1e-8, max_iter: int=100):
    """ Performs Mueller's method element-wise """
    for i in range(max_iter):
        fval2 = f(x2)
        max_val = np.abs(fval2)
        max_res = np.abs(x2 - x1)
        if max_res < tol or max_val < tol:
            logger.debug(f"Mueller's method converged in {i+1}/{max_iter} steps, last MAX(|step|) = {max_res}, MAX(|fval|) = {max_val}")
            return x2, True
        
        fval0 = f(x0)
        fval1 = f(x1)
        
        # TODO assert h0 nonzero h1 nonzero
        h0 = x1 - x0
        h1 = x2 - x1
        delta0 = (fval1 - fval0) / h0
        delta1 = (fval2 - fval1) / h1

        # solve parabola
        a = (delta1 - delta0) / (h1 + h0)
        b = a*h1 + delta1
        c = fval2

        # reassign values
        x0 = x1
        x1 = x2
        x2 = x1 - 2*c / (b + np.sign(b) * np.sqrt( np.power(b, 2) - 4*a*c ))
    
    logger.warning(f"Mueller's method did not converged in {max_iter} steps, last MAX(|res|) = {max_res}")
    return x2, False

def mueller(f: Callable, x0: float | npt.NDArray, x1: float | npt.NDArray=None, x2: float | npt.NDArray=None, tol: float=1e-6, max_iter: int=100) -> tuple[float | complex | npt.NDArray, bool]:
    """Attempt to solve :math:`f(x) = 0` via Müller's method.

  Uses a quadratic interpolating polynomial through three successive iterates.
  Scalar inputs are not yet implemented; array inputs are solved element-wise.

    Parameters
    ----------
    f : callable
        Function to find a root of.
    x0 : float or ndarray
        First initial guess.
    x1 : float or ndarray, optional
        Second initial guess. If ``None``, obtained by perturbing ``x0``.
    x2 : float or ndarray, optional
        Third initial guess. If ``None``, obtained by one secant update.
    tol : float, optional
        Stopping tolerance. Default is 1e-6.
    max_iter : int, optional
        Maximum number of iterations. Default is 100.

    Returns
    -------
    x : ndarray
        Approximate root(s).
    converged : bool
        ``True`` if the method converged within ``max_iter`` iterations.

    Raises
    ------
    NotImplementedError
        If ``x0`` is a scalar.
    """
    if isinstance(x0, (float, int)):
        #return _newton_scalar(f, f_prime, x0, m=m, tol=tol, max_iter=max_iter)
        raise NotImplementedError
    
    elif isinstance(x0, np.ndarray):
        # TODO, check all x0 same arrays
        # TODO, check x1-x2 > 0, x2-x1 > 0

        if x1 is None:
            # x1 is undefined -> obtain by perturbation
            x1 = x0 + np.maximum(np.finfo(x0.dtype).eps * np.abs(x0), max(10e-8, 2*tol))
        
        if x2 is None:
            # x2 is undefined -> use one secant update to obtain x2
            x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        
        return _mueller_array(f, x0, x1, x2, tol=tol, max_iter=max_iter)
    else:
        raise NotImplementedError(f"Not implemented")
