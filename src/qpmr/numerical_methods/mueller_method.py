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
    """ TODO
    
    """
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

def mueller(f: Callable, x0: float | npt.NDArray, x1: float | npt.NDArray=None, x2: float | npt.NDArray=None, tol: float=1e-6, max_iter: int=100) -> float:
    """ Attemts to approximate solution of f(x)=0 via Newton's method

    Args:
        f (callable): function f(x)
        f_prime (callable): derivative of f(x)
        x0 (float | array): initial guess for solution of f(x)=0
        m (float | array): multiplicity to recover quadratic convergence,
            default 1.0
        tol (float): positive number which defines stopping criteria
            |f(x)| < eps, default 1e-6
        max_iter (int): maximum number of iterations, default 100

    Returns:
        tuple containing

        - x (ndarray): roots with increased precission
        - converged (bool): True if successful, False otherwise

    
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
