"""
QPmR v2 implementation
"""

import logging
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def grid_size_heuristic(region, *args, **kwargs) -> float:
    r = (region[1] - region[0]) * (region[3] - region[2]) / 1000.
    logger.debug(f"Grid size not specified, setting as ds={r} (solved by heuristic)")
    return r

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
    e = kwargs.get("e", 1e-6)
    ds = kwargs.get("ds", grid_size_heuristic(region))

    assert isinstance(e, float) and e > 0.0, "error 'e' numerical accuracy"
    assert isinstance(ds, float) and ds > 0.0, "error 'ds' grid stepsize"
    dtype = kwargs.get("dtype", None) # TODO not used

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
        _memory *= complex_grid # _memory = np.multiply(_memory, complex_grid)

    func_value_real = np.real(func_value)
    func_value_imag = np.imag(func_value)

    ## finding contours
    quad_contour = plt.contour(real_range, imag_range, func_value_real, levels=[0])
    segments_real = quad_contour.allsegs
    quad_contour = plt.contour(real_range, imag_range, func_value_imag, levels=[0])
    segments_imag = quad_contour.allsegs

    # detecting intersection points
    print(segments_real)
    print(type(segments_real))
    # continue on line 332

    return func_value, complex_grid


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    region = [-10, 2, 0, 30]
    delays = np.array([0.0, 1.0])
    coefs = np.array([[0, 1],[1, 0]])

    value, complex_grid = qpmr(region, coefs, delays)

    def h(s):
        return s + np.exp(-s)

    if False:
        plt.figure()
        plt.contour(np.real(complex_grid), np.imag(complex_grid), np.real(value), levels=[0], colors='blue')
        plt.contour(np.real(complex_grid), np.imag(complex_grid), np.imag(value), levels=[0], colors='green')

        plt.figure()
        plt.contour(np.real(complex_grid), np.imag(complex_grid), np.real(h(complex_grid)), levels=[0], colors='blue')
        plt.contour(np.real(complex_grid), np.imag(complex_grid), np.imag(h(complex_grid)), levels=[0], colors='green')

        plt.show()


