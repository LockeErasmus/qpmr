"""
Secant method
-------------
"""

import logging
from typing import Callable

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

def secant(func: Callable, x0, x1=None, tol: float=1e-8, max_iter: int=100) -> tuple[npt.NDArray, bool]:
    """Secant method for vectorized root refinement.

    Parameters
    ----------
    func : callable
        Vectorized complex function.
    x0 : ndarray
        Primary initial guesses.
    x1 : ndarray, optional
        Secondary initial guesses. If ``None``, obtained by perturbing ``x0``.
    tol : float, optional
        Convergence tolerance. Default is 1e-8.
    max_iter : int, optional
        Maximum iterations. Default is 100.

    Returns
    -------
    x : ndarray
        Refined roots.
    converged : bool
        Whether the method converged.
    """
    x = np.copy(x0)
    eval_counter = 0
    if x1 is None:
        logger.debug(f"Initial x1 not provided and therefore will be solved by heuristic")
        x1 = x + 2 * tol
        x = x - 2 * tol
    
    for i in range(max_iter):
        x2 = x1 - func(x1) * (x1 - x) / (func(x1) - func(x0))
        eval_counter += 2
        x, x1 = x1, x2
        max_res = np.max(np.abs(x-x1))
        if max_res <= tol:
            logger.debug(f"Secant converged in {i+1}/{max_iter} steps| func evals={eval_counter}, last MAX(|res|) = {max_res}")
            converged = True
            break    
    if i == max_iter - 1:
        logger.warning(f"Secant did not converged in {max_iter} steps, last MAX(|res|) = {max_res}")
        converged = False

    return x1, converged
