"""
Numerical methods for increasing roots precission
"""
import logging
from typing import Callable

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

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

def secant(func: Callable, x0, x1=None, tolerance: float=1e-8, max_iterations: int=100) -> tuple[npt.NDArray, bool]:
    """ Secant method
    
    Args:
        func (callable): vectorized quasi-polynomical function which maps Complex -> Complex
        x0 (ndarray): initial guess 0 for roots
        x1 (ndarray): initial guess 1 for roots, default None
        tolerance (float): required tolerance, default 1e-7
        max_iterations (int): maximum iterations, default 100

    Returns:
        tuple containing

        - x (ndarray): roots with increased precission
        - converged (bool): True if successful, False otherwise
    """
    x = np.copy(x0)
    eval_counter = 0
    if x1 is None:
        logger.debug(f"Initial x1 not provided and therefore will be solved by heuristic")
        x1 = x + 2 * tolerance * (1. + 1j)
        x = x - 2 * tolerance * (1. + 1j)
    
    for i in range(max_iterations):
        x2 = x1 - func(x1) * (x1 - x) / (func(x1) - func(x0))
        eval_counter += 2
        x, x1 = x1, x2
        max_res = np.max(np.abs(x-x1))
        if max_res <= tolerance:
            logger.debug(f"Secant converged in {i+1}/{max_iterations} steps| func evals={eval_counter}, last MAX(|res|) = {max_res}")
            converged = True
            break
    
    if i == max_iterations - 1:
        logger.warning(f"Secant did not converged in {max_iterations} steps, last MAX(|res|) = {max_res}")
        converged = False

    return x1, converged
