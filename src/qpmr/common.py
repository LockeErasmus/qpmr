"""
Compilation of mathematical utility functions
"""

import logging

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

def find_crossings(x: npt.NDArray, y: npt.NDArray, remove_consequent: bool=True, interpolate: bool=False) -> npt.NDArray:
    """ Finds 0-level crossings by checking consequent difference in signs
    
    Args:
        x (array): complex vector representing real contour
        y (array): real vector representing Im( h(z) )
        TODO

    """
    mask = (np.abs(np.diff(np.sign(y))).astype(bool) # corssings when sign of Im changes
            | np.abs(np.diff(np.sign(x.imag))).astype(bool)) # crossings of real contour with real axis
    if remove_consequent:
        # handles sign sequences of altering signs "+-+" or "-+-"
        mask_consequent = mask[:-1] & mask[1:]
        mask[1:] = np.bitwise_xor(mask[1:], mask_consequent)
    if interpolate: # linear interpolation
        return x[:-1][mask] + np.diff(x)[mask]/(np.abs(y[1:][mask]/y[:-1][mask])+1)
    else: # half of the interval, note that this is much better in practice, because values of extremely large modulus
        return 0.5 * ( x[:-1][mask] + x[1:][mask])