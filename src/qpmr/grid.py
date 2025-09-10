"""
Grid
----
Notes:
    ---
"""

import numpy as np
import numpy.typing as npt

def grid_size_heuristic(region: tuple[float, float, float, float], coefs: npt.NDArray, delays: npt.NDArray) -> float:
    """ Grid size heuristic original 2009 """
    alpha_max = np.max(delays) if delays.size > 0 else 0. # biggest delay
    if alpha_max == 0.:
        return (region[1] - region[0]) * (region[3] - region[2]) / 1000.
    else:
        return np.pi / 10 / alpha_max

def complex_grid_rect(region: tuple, ds: float):
    """ Creates complex grid from rectangular region """
    
    bmin, bmax, wmin, wmax = region
    real_range = np.arange(bmin, bmax, ds)
    imag_range = np.arange(wmin, wmax, ds)
    return 1j*imag_range.reshape(-1, 1) + real_range
