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

def newton(f: Callable, df: Callable, x0: float, eps: float=1e-6, max_iter: int=100) -> float:
    """ Attemts to approximate solution of f(x)=0 via Newton's method

    Args:
        f (callable): function f(x)
        df (callable): derivative of f(x)
        x0 (float): initial guess for solution of f(x)=0
        eps (float): positive number which defines stopping criteria
            |f(x)| < eps, default 1e-6
        max_iter (int): maximum number of iterations, default 100

    Returns:
        xn (float): approximate solution of f(x)=0, None if `max_iter` reached
            and condition |f(x)| < eps not met or encountered df(x)=0 zero
            derivative
    """
    xn = x0
    for n in range(0, max_iter):
        fxn = f(xn)
        if abs(fxn) < eps:
            logger.debug(f"Solution{xn=} found in {n} iterations")
        dfxn = df(xn)
        if dfxn == 0:
            logger.debug(f"Encountered df(xn) = 0, returning xn=None")
            return None
        xn = xn - fxn/dfxn
    logger.debug(f"Exceeded {max_iter=}, returning xn=None")
    return None

def numerical_newton(func: Callable, x0: npt.NDArray, tolerance: float=1e-7, max_iterations: int=100) -> tuple[npt.NDArray, bool]:
    """ Numerical newton method
    
    Implementation of original numerical method from QPmR v2

    Args:
        func (callable): vectorized quasi-polynomical function which maps Complex -> Complex
        x0 (ndarray): initial guesses for roots
        tolerance (float): required tolerance, default 1e-7
        max_iterations (int): maximum iterations, default 100

    Returns:
        tuple containing

        - x (ndarray): roots with increased precission
        - converged (bool): True if successful, False otherwise
    """
    x = np.copy(x0)
    eps = tolerance * 0.1 # epsilon TODO why 0.1 - just taken from original implementation
    for i in range(max_iterations):
        val = func(x)
        dfunc = (func(x-eps) - func(x+eps) + 1j*func(x+1j*eps) - 1j*func(x-1j*eps)) / 4. / eps
        step = val / dfunc
        max_res = np.max(np.abs(step))
        x += step
        if max_res <= tolerance:
            logger.debug(f"Numerical Newton converged in {i+1}/{max_iterations} steps, last MAX(|res|) = {max_res}")
            converged = True
            break
    if i == max_iterations - 1:
        logger.warning(f"Numerical Newton did not converged in {max_iterations} steps, last MAX(|res|) = {max_res}")
        converged = False
    return x, converged