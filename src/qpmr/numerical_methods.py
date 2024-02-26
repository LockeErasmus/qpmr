"""
Numerical methods for increasing roots precission
"""
import logging
from typing import Callable

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

def numerical_newton(func: Callable, x0: npt.NDArray, tolerance: float=1e-7, max_iterations: int=100):
    """ Implementation of original numerical method from QPmR v2    
    """
    x = np.copy(x0)
    eps = tolerance * 1e-2 # epsilon TODO why 0.01 - just taken from original implementation
    for i in range(max_iterations):
        val = func(x)
        dfunc = (func(x-eps) - func(x+eps) + 1j*func(x+1j*eps) - 1j*func(x-1j*eps)) / 4. / eps
        step = val / dfunc
        max_res = np.max(np.abs(step))
        x += step
        if max_res <= 0.1 * tolerance: # TODO why 0.1 - just taken from original implementation
            logger.debug(f"Numerical Newton converged in {i+1}/{max_iterations} steps, last MAX(|res|) = {max_res}")
            converged = True
            break
    if i == max_iterations - 1:
        # TODO also add warnings.warn(.) ?
        logger.warning(f"Numerical Newton did not converged in {max_iterations} steps, last MAX(|res|) = {max_res}")
        converged = False
    return x, converged

def secant(x0, x1=None, tolerance=1e-7, max_iterations=20):
    raise NotImplementedError("Secant method is yet to be implemented")
