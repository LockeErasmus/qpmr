"""
"""
import logging
from typing import Callable


import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

def argument_principle(func: Callable, region, ds, eps) -> float:
    """
    func ...
    region ...
    ds ... step for grid (region)
    eps ... step for calculating numerical derivative of `func`
    """
    
    # enlarge the region by ds to each side
    reg = [region[0]-ds, region[1]+ds, region[2]-ds, region[3]+ds]

    n_steps_real = int((reg[1] - reg[0]) / ds) + 1
    linspace_real = np.linspace(reg[0], reg[1], n_steps_real)
    step_real = (reg[1] - reg[0]) / (n_steps_real - 1)

    n_steps_imag = int((reg[3] - reg[2]) / ds) + 1
    linspace_imag = np.linspace(reg[2] - ds, reg[3] + ds, n_steps_imag)
    step_imag = (reg[3] - reg[2] + 2*ds) / (n_steps_imag - 1)

    contour = np.r_[linspace_real + 1j*reg[2],
                    reg[1] + 1j*linspace_imag,
                    np.flip(linspace_real) + 1j*reg[3],
                    reg[0] + 1j*np.flip(linspace_imag)]
    contour_steps = np.r_[np.full(shape=(n_steps_real,), fill_value=step_real),
                          np.full(shape=(n_steps_imag,), fill_value=1j*step_imag),
                          np.full(shape=(n_steps_real,), fill_value=-step_real),
                          np.full(shape=(n_steps_imag,), fill_value=-1j*step_imag)]

    # calculate d func / dz
    func_value = func(contour)
    func_value_derivative = (func(contour - eps)
                             - func(contour + eps)
                             + 1j*func(contour +1j*eps)
                             - 1j*func(contour -1j*eps)) / 4. / eps

    # use argument principle
    n_raw = np.abs(np.real(1 / (2 * np.pi * 1j) * np.sum(func_value_derivative / func_value * contour_steps)))

    logger.debug(f"Using argument principle, contour integral = {n_raw}")
    # to round or not to round ??? TODO decide
    return np.round(n_raw)