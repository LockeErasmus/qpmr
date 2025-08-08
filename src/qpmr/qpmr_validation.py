"""
Functions for validating QPmR inputs
------------------------------------

"""

import numpy as np

def validate_region(region) -> tuple[float, float, float, float]:
    # Acceptable types: tuple, list, np.ndarray
    if not isinstance(region, (tuple, list, np.ndarray)):
        raise TypeError("region must be a tuple, list, or NumPy array of 4 floats")
    
    if len(region) != 4:
        raise ValueError("region must have exactly 4 elements")

    # If numpy array: check dtype and convert elements to float for consistency
    if isinstance(region, np.ndarray):
        if region.dtype.kind not in {'f', 'i'}:  # float or int allowed (can be cast)
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
    
    return tuple(floats)  # Return a consistent type, e.g., tuple
