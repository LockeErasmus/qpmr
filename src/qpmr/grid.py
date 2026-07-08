r"""
Grid utilities
==============

Heuristics and helpers for building complex-plane grids used in spectrum
mapping and QPmR.
"""

import numpy as np
import numpy.typing as npt


def grid_size_heuristic(region: tuple[float, float, float, float], coefs: npt.NDArray, delays: npt.NDArray) -> float:
    """Propose a grid step size for spectrum mapping.

    Uses the original QPmR (2009) heuristic based on the largest delay, or
    scales with region area when no delays are present.

    Parameters
    ----------
    region : tuple of float
        Rectangular region ``(Re_min, Re_max, Im_min, Im_max)``.
    coefs : ndarray
        Matrix of polynomial coefficients. Each row represents the coefficients
        corresponding to a specific delay.
    delays : ndarray
        Vector of delays associated with each row in ``coefs``.

    Returns
    -------
    ds : float
        Recommended grid step size.
    """
    alpha_max = np.max(delays) if delays.size > 0 else 0.
    if alpha_max == 0.:
        return (region[1] - region[0]) * (region[3] - region[2]) / 1000.
    else:
        return np.pi / 10 / alpha_max


def complex_grid_rect(region: tuple, ds: float):
    """Build a 2D complex grid over a rectangular region.

    Parameters
    ----------
    region : tuple of float
        Rectangular region ``(Re_min, Re_max, Im_min, Im_max)``.
    ds : float
        Grid step along real and imaginary axes.

    Returns
    -------
    grid : ndarray
        2D array of complex points covering the rectangle.
    """
    bmin, bmax, wmin, wmax = region
    real_range = np.arange(bmin, bmax, ds)
    imag_range = np.arange(wmin, wmax, ds)
    return 1j*imag_range.reshape(-1, 1) + real_range
