"""
QPmR v2 implementation
"""
import logging
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from .numerical_methods import numerical_newton

logger = logging.getLogger(__name__)


def grid_size_heuristic(region, *args, **kwargs) -> float:
    r = (region[1] - region[0]) * (region[3] - region[2]) / 1000.
    logger.debug(f"Grid size not specified, setting as ds={r} (solved by heuristic)")
    return r

def find_roots(x, y):
    """ Finds 0-level crossings by checking consequent difference in signs, ie
    
    s[k] is True if sign(y[k]), sign(y[k+1]) is either `+-` or `-+`

    """
    s = np.abs(np.diff(np.sign(y))).astype(bool)
    return x[:-1][s] + np.diff(x)[s]/(np.abs(y[1:][s]/y[:-1][s])+1)

def create_vector_callable(coefs, delays):
    degree, num_delays = np.shape(coefs)
    def func(z):
        shape = np.shape(z)
        _memory = np.ones(shape, dtype=z.dtype)
        delay_terms = np.exp(
            (np.tile(z, (len(delays), 1))
             * (-delays[:, np.newaxis]))
        )
        val = np.zeros(shape, dtype=z.dtype)
        for d in range(degree):
            val += np.sum(delay_terms * _memory[np.newaxis, :] * coefs[:, d][:, np.newaxis], axis=0)
            _memory *= z
        return val
    return func

def qpmr(region: list, coefs, delays, **kwargs):
    """

    Args:
        region: TODO
        coefs: TODO
        delays: TODO

        **kwargs:
            e (float) - computation accuracy, default = 1e-6
            ds (float) - grid step, default obtained by heuristic
    
    """

    assert len(region) == 4, "TODO"

    # TODO assert coefs dimensions, match delays dimensions
    # TODO assert degree >= 0, meaning coefs has at least 1 column
    
    # defaults
    logger.debug(f"defined kwargs {kwargs}")
    e = kwargs.get("e", 1e-6)
    ds = kwargs.get("ds", None)
    if not ds:
        ds = grid_size_heuristic(region)
    
    # roots precission
    max_iterations = kwargs.get("max_iterations", 10)


    assert isinstance(e, float) and e > 0.0, "error 'e' numerical accuracy"
    assert isinstance(ds, float) and ds > 0.0, "error 'ds' grid stepsize"

    # extend region and create meshgrid (original algorithm) -> TODO move to function
    bmin=region[0] - 3*ds
    bmax=region[1] + 3*ds
    wmin=region[2] - 3*ds
    wmax=region[3] + 3*ds
    real_range = np.arange(bmin, bmax, ds)
    imag_range = np.arange(wmin, wmax, ds)
    complex_grid = 1j*imag_range.reshape(-1, 1) + real_range

    # values of function -> TODO move to separate function
    degree, num_delays = coefs.shape # TODO variables keep, move up and rework
    func_value = np.zeros(complex_grid.shape, dtype=complex_grid.dtype)
    _memory = np.ones(complex_grid.shape, dtype=complex_grid.dtype) # x*x*..*x much faster than np.power(x, N)
    ## prepare exp(-s*tau)
    # TODO delays has to be (num_delays,) vector
    delay_grid = np.exp(
        (np.tile(complex_grid, (len(delays), 1, 1)) # N times complex grid, [0,:,:] is complex grid for delay1 etc.
         * (-delays[:, np.newaxis, np.newaxis])) # power of broadcasting [delay1, delay2, ..., delayN]
    )
    for d in range(degree):       
        func_value += np.sum(delay_grid * _memory[np.newaxis, :, :] * coefs[:, d][:, np.newaxis, np.newaxis], axis=0)
        _memory *= complex_grid 

    func_value_real = np.real(func_value)
    # func_value_imag = np.imag(func_value) # not needed in original algorithm

    ## finding contours
    quad_contour = plt.contour(real_range, imag_range, func_value_real, levels=[0])
    zero_level_contours = quad_contour.allsegs[0] # only level-0 polygons
    # quad_contour = plt.contour(real_range, imag_range, func_value_imag, levels=[0])
    # segments_imag = quad_contour.allsegs

    if zero_level_contours is None: # TODO no raise error but return None?
        raise ValueError("No 0-level contours found")
    
    # detecting intersection points
    roots = []
    for polygon in zero_level_contours:
        polygon_complex = polygon[:,0] + 1j*polygon[:,1]
        
        # calculate value for all of these points
        delay_terms = np.exp(
            (np.tile(polygon_complex, (len(delays), 1))
             * (-delays[:, np.newaxis]))
        )
        polygon_func_value = np.zeros(polygon_complex.shape, dtype=polygon_complex.dtype)
        _memory = np.ones(polygon_complex.shape, dtype=polygon_complex.dtype)
        for d in range(degree):       
            polygon_func_value += np.sum(delay_terms * _memory[np.newaxis, :] * coefs[:, d][:, np.newaxis], axis=0)
            _memory *= polygon_complex # _memory = np.multiply(_memory, complex_grid)
        
        # find all intersections
        polygon_func_imag = np.imag(polygon_func_value)
        roots.append(find_roots(polygon_complex, polygon_func_imag))
    roots = np.hstack(roots)

    # apply numerical method to increase precission
    func = create_vector_callable(coefs, delays)
    roots = numerical_newton(func, roots) # TODO inplace=True?

    # filter out roots that are not in predefined region
    mask = ((roots.real >= region[0]) & (roots.real <= region[1]) # Re bounds
            & (roots.imag >= region[2]) & (roots.imag <= region[3])) # Im bounds
    roots = roots[mask]

    # TODO - decide what to do with metadata --> dataclass ?
    metadata = {
        "func_value": func_value,
        "complex_grid": complex_grid,
    }
    return roots, metadata
