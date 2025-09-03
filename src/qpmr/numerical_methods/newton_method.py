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

def _newton_scalar(f: Callable, f_prime: Callable, x0: float | complex, m: float=1.0, tol: float=1e-8, max_iter: int=100):
    """ TODO
    
    """
    x = x0
    for i in range(max_iter):
        fval = f(x)
        fval_abs = abs(fval)
        if fval_abs < tol:
            logger.debug(f"Newton[scalar] converged in {i+1}/{max_iter} steps, |fval| = {fval_abs}")
            return x, True
        
        f_val_prime = f_prime(x)
        if abs(f_val_prime) < 1e-14:
            logger.warning(f"NNewton[scalar] did not converge: Derivative too small (1e-14) in {i+1}/{max_iter} steps")
            return x, False
        
        step = m * (fval / f_val_prime)
        step_abs = abs(step)
        x -= step
        if step_abs < tol:
            logger.debug(f"Newton[scalar] converged in {i+1}/{max_iter} steps, |step|= {step_abs} |fval| = {fval_abs}")
            return x, True

    logger.warning(f"Newton[scalar] did not converge in {max_iter} steps, last |step|= {step_abs} |fval| = {fval_abs}")
    return x, False

def _newton_array(f: Callable, f_prime: Callable, x0: npt.NDArray, m: npt.NDArray=1.0, tol: float=1e-8, max_iter: int=100):
    """ Performs Newton method element-wise

    Args:
        m (): multiplicity of roots
    
    Stopping conditions:
        1. `tol` is smaller then the largest `step`
    
    TODO
    """
    x = np.copy(x0)
    for i in range(max_iter):
        f_val = f(x)
        f_prime_val = f_prime(x)

        if np.any(np.abs(f_prime_val) < 1e-14):
            logger.warning(f"Numerical Newton did not converged: in {i+1}/{max_iter} steps Derivative too small (< 1e-14)")
            return x, False

        step = m * (f_val / f_prime_val)
        max_res = np.max(np.abs(step))
        max_val = np.max(np.abs(f_val))
        x -= step
        if max_res < tol or max_val < tol:
            logger.debug(f"Numerical Newton converged in {i+1}/{max_iter} steps, last MAX(|step|) = {max_res}, MAX(|fval|) = {max_val}")
            return x, True
    
    logger.warning(f"Numerical Newton did not converged in {max_iter} steps, last MAX(|res|) = {max_res}")
    return x, False

def newton(f: Callable, f_prime: Callable, x0: float | npt.NDArray, m: float | npt.NDArray=1.0, tol: float=1e-6, max_iter: int=100) -> float:
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
        return _newton_scalar(f, f_prime, x0, m=m, tol=tol, max_iter=max_iter)
    elif isinstance(x0, np.ndarray):
        return _newton_array(f, f_prime, x0, m=m, tol=tol, max_iter=max_iter)
    else:
        raise NotImplementedError(f"Not implemented")

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
    eps = tolerance * 0.1 # epsilon TODO why 0.1 - just taken from original implementation
    f_prime = lambda s: (func(s + eps) - func(s - eps)) / 2. / eps


    # f_prime = lambda s: (func(s + eps) - func(s - eps) / 4. / eps + )

    return _newton_array(func, f_prime, x0, tol=tolerance, max_iter=max_iterations)
