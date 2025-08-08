"""
Functions for validating QPmR inputs
------------------------------------
"""

import numpy as np
import numpy.typing as npt

def validate_region(region) -> tuple[float, float, float, float]:
    """ Validates rectangular complex region definition and returns as tuple """

    if not isinstance(region, (tuple, list, np.ndarray)):
        raise TypeError("region must be a tuple, list, or NumPy array of 4 floats")
    
    if len(region) != 4:
        raise ValueError("region must have exactly 4 elements")
    
    # If numpy array: check dtype and convert elements to float for consistency
    if isinstance(region, np.ndarray):
        if region.dtype.kind not in {'f', 'i'}:  # float or int allowed
            raise TypeError("NumPy array must be of float or int dtype")
        if not np.issubdtype(region.dtype, np.number):
            raise TypeError("NumPy array must contain numeric values only")
        floats = region.astype(float)
    else:
        # For list/tuple: check each element is float (or castable to float)
        try:
            floats = [float(x) for x in region]
        except (ValueError, TypeError):
            raise TypeError("All elements in region must be floats or castable to float")

    # Semantic check
    if floats[1] <= floats[0] or floats[3] <= floats[2]:
        raise ValueError("Invalid region bounds: must satisfy region[1] > region[0] and region[3] > region[2]")
    
    return tuple(floats)

def validate_qp(coefs: npt.NDArray[np.float64], delays: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """ Validates quasipolynomial (coefs, delays) definition and returns as tuple """
    # check coefs
    if not isinstance(coefs, np.ndarray):
        raise TypeError("coefs has to be a NumPy ndarray")
    
    if coefs.ndim != 2:
        raise ValueError("coefs has to be 2D array")
    
    if coefs.size == 0:
        raise ValueError("coefs must be non-empty.")
    
    # check delays
    if not isinstance(delays, np.ndarray):
        raise TypeError("delays has to be a NumPy ndarray")
    
    if delays.ndim != 1:
        raise ValueError("delays has to be 1D array")
    
    if delays.size == 0:
        raise ValueError("delays must be non-empty.")
    
    # check compatible first dimension
    if coefs.shape[0] != delays.shape[0]:
        raise ValueError(f"coefs and delays has to have shapes N x M and N, respectively")

    return coefs, delays
