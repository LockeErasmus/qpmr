"""
Newton's method
---------------
Notes:
    1. vectorized version easily obtainable
"""
from typing import Callable
import logging

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
