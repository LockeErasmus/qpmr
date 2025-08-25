"""
Argument Principle implementation
---------------------------------
TODO:
    1. implement different integration method
"""
import logging
from typing import Callable

import numpy as np
import numpy.typing as npt

from .quasipoly.core import _eval_array

logger = logging.getLogger(__name__)

def _discretize_rectangular_boundary(bmin: float, bmax: float, wmin: float, wmax: float, ds: float):
    """ Discretizes rectangular boundary contour into long complex vector

    Note that first and last numbers are the same.

    Args:
        TODO
    Returns:
        TODO
    """
    n_steps_real = int((bmax - bmin) / ds) + 1
    linspace_real = np.linspace(bmin, bmax, n_steps_real)
    step_real = (bmax - bmin) / (n_steps_real - 1)

    n_steps_imag = int((wmax - wmin) / ds) + 1
    linspace_imag = np.linspace(wmin - ds, wmax + ds, n_steps_imag)
    step_imag = (wmax - wmin + 2*ds) / (n_steps_imag - 1)

    contour = np.r_[linspace_real + 1j*wmin,
                    bmax + 1j*linspace_imag,
                    np.flip(linspace_real) + 1j*wmax,
                    bmin + 1j*np.flip(linspace_imag)]
    contour_steps = np.r_[np.full(shape=(n_steps_real,), fill_value=step_real),
                          np.full(shape=(n_steps_imag,), fill_value=1j*step_imag),
                          np.full(shape=(n_steps_real,), fill_value=-step_real),
                          np.full(shape=(n_steps_imag,), fill_value=-1j*step_imag)]
    return contour, contour_steps

def _check_no_zero_boundary(coefs: npt.NDArray, delays: npt.NDArray, region: tuple[float, float, float, float], **kwargs):
    """ Checks that no zeros are on boundary curve
    
    """
    raise NotImplementedError(".")


def _argument_principle_circle(f: Callable, z0: complex, radius: float, df: Callable=None, num: int=1000):
    # solve callable
    # parametrize circle contour
    t = np.linspace(0., 2*np.pi, num=num+1, endpoint=False)
    contour = z0 + radius * np.exp(1j * t)

    vals = f(contour)
    if df is None:
        eps = 1e-8
        dvals = (f(contour - eps)
                - f(contour + eps)
                + 1j*f(contour +1j*eps)
                - 1j*f(contour -1j*eps)) / 4. / eps
    else:
        dvals = df(contour)

    n_raw = np.abs(np.real(1 / (2 * np.pi * 1j) * np.sum( (np.roll(contour, -1) - contour) * dvals / vals )))
    n = np.round(n_raw)
    logger.debug(f"Using argument principle (CIRCLE({z0=}, {radius=})), contour integral = {n_raw} | rounded to {n}")
    return n

    

def argument_principle(func: Callable, region: tuple[float, float, float, float],
                       ds: float, eps: float) -> float:
    """ Evaluates number of roots in given rectangular region via argument
    principle

    Args:
        func (Callable): vector-suitable callable for quassipolynomial
        region (list of `float`): region in complex plane
        ds (float): grid stepsize
        eps (float): step for finite difference method

    Returns:
        n (float): rounded number of complex roots in region based on numerical
            integration and argument principle
    """
    # prepare contour path
    contour, contour_steps = _discretize_rectangular_boundary(*region, ds=ds)

    # calculate d func / dz
    func_value = func(contour)
    func_value_derivative = (func(contour - eps)
                             - func(contour + eps)
                             + 1j*func(contour +1j*eps)
                             - 1j*func(contour -1j*eps)) / 4. / eps

    # use argument principle and round
    n_raw = np.abs(np.real(1 / (2 * np.pi * 1j) * np.sum(func_value_derivative / func_value * contour_steps)))
    n = np.round(n_raw)
    logger.debug(f"Using argument principle, contour integral = {n_raw} | rounded to {n}")

    return n