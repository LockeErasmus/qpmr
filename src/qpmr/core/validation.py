"""
Functions for validating QPmR inputs
------------------------------------
"""

import numpy as np
import numpy.typing as npt


def validate_region(region) -> tuple[float, float, float, float]:
    """Validate and normalize a rectangular complex-plane region.

    Parameters
    ----------
    region : tuple, list, or ndarray
        Four numeric bounds ``(Re_min, Re_max, Im_min, Im_max)``.

    Returns
    -------
    bounds : tuple of float
        Validated region bounds.

    Raises
    ------
    TypeError
        If ``region`` is not a sequence of numeric values.
    ValueError
        If length is not 4 or bounds are not strictly ordered.
    """
    if not isinstance(region, (tuple, list, np.ndarray)):
        raise TypeError("region must be a tuple, list, or NumPy array of 4 floats")

    if len(region) != 4:
        raise ValueError("region must have exactly 4 elements")

    if isinstance(region, np.ndarray):
        if region.dtype.kind not in {'f', 'i'}:
            raise TypeError("NumPy array must be of float or int dtype")
        if not np.issubdtype(region.dtype, np.number):
            raise TypeError("NumPy array must contain numeric values only")
        floats = region.astype(float)
    else:
        try:
            floats = [float(x) for x in region]
        except (ValueError, TypeError):
            raise TypeError("All elements in region must be floats or castable to float")

    if floats[1] <= floats[0] or floats[3] <= floats[2]:
        raise ValueError("Invalid region bounds: must satisfy region[1] > region[0] and region[3] > region[2]")

    return tuple(floats)


def validate_qp(coefs: npt.NDArray[np.float64], delays: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Validate quasi-polynomial coefficient and delay arrays.

    Parameters
    ----------
    coefs : ndarray
        2D coefficient matrix (non-empty).
    delays : ndarray
        1D delay vector (non-empty), same length as rows of ``coefs``.

    Returns
    -------
    coefs : ndarray
        Validated coefficient matrix.
    delays : ndarray
        Validated delay vector.

    Raises
    ------
    TypeError
        If inputs are not NumPy arrays of the expected dimensionality.
    ValueError
        If shapes are incompatible or arrays are empty.
    """
    if not isinstance(coefs, np.ndarray):
        raise TypeError("coefs has to be a NumPy ndarray")

    if coefs.ndim != 2:
        raise ValueError("coefs has to be 2D array")

    if coefs.size == 0:
        raise ValueError("coefs must be non-empty.")

    if not isinstance(delays, np.ndarray):
        raise TypeError("delays has to be a NumPy ndarray")

    if delays.ndim != 1:
        raise ValueError("delays has to be 1D array")

    if delays.size == 0:
        raise ValueError("delays must be non-empty.")

    if coefs.shape[0] != delays.shape[0]:
        raise ValueError(f"coefs and delays has to have shapes N x M and N, respectively")

    return coefs, delays
