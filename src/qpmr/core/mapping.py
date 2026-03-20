r"""
Spectrum mapping algorithm
--------------------------

"""

from typing import Callable

import contourpy
import numpy as np
import numpy.typing as npt

def _find_crossings(x: npt.NDArray, y: npt.NDArray,
                    remove_consequent: bool=True,
                    interpolate: bool=False) -> npt.NDArray:
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

def _spectrum_mapping(f: Callable, re_range: npt.NDArray, im_range: npt.NDArray):
    """ Spectrum mapping algorithm """
    contour_generator = contourpy.contour_generator(
        x=re_range,
        y=im_range,
        z=f( 1j*im_range.reshape(-1, 1) + re_range )
    )
    roots = [np.empty(shape=(0,), dtype=np.complex128)]
    for c in contour_generator.lines(0.0):
        # detect intersection points
        re_contour = 1j * c[:, 1] + c[:, 0]
        crossings = _find_crossings(
            re_contour,
            np.imag( f(re_contour) ),
            remove_consequent=True,
            interpolate=True, # TODO, is it necessary?
        )
        if crossings.size:
            roots.append(crossings)
    return np.hstack(roots)

def spectrum_mapping(f: Callable, rectangle: tuple[float, float, float, float], ds: float):
    raise NotImplementedError("...")