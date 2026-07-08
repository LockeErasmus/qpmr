r"""
Spectrum mapping algorithm
--------------------------

Contour-based root localization used by QPmR.
"""

from typing import Callable

import contourpy
import numpy as np
import numpy.typing as npt

from . import quasipolynomial


def _find_crossings(x: npt.NDArray, y: npt.NDArray,
                    remove_consequent: bool=True,
                    interpolate: bool=False) -> npt.NDArray:
    """Find zero-level crossings along a contour segment.

    Parameters
    ----------
    x : ndarray
        Complex points along the contour.
    y : ndarray
        Real part of the mapped function values (sign used for crossing).
    remove_consequent : bool, optional
        Merge consecutive crossing flags. Default is ``True``.
    interpolate : bool, optional
        If ``True``, linearly interpolate crossing positions.

    Returns
    -------
    crossings : ndarray
        Complex crossing locations.
    """
    mask = (np.abs(np.diff(np.sign(y))).astype(bool)
            | np.abs(np.diff(np.sign(x.imag))).astype(bool))
    if remove_consequent:
        mask_consequent = mask[:-1] & mask[1:]
        mask[1:] = np.bitwise_xor(mask[1:], mask_consequent)
    if interpolate:
        return x[:-1][mask] + np.diff(x)[mask]/(np.abs(y[1:][mask]/y[:-1][mask])+1)
    else:
        return 0.5 * ( x[:-1][mask] + x[1:][mask])


def _spectrum_mapping(f: Callable, re_range: npt.NDArray, im_range: npt.NDArray):
    """Apply spectrum mapping on a rectangular grid.

    Parameters
    ----------
    f : callable
        Complex function evaluated on a meshgrid.
    re_range : ndarray
        Real-axis sample points.
    im_range : ndarray
        Imaginary-axis sample points.

    Returns
    -------
    roots : ndarray
        Approximate root locations from zero-level contour crossings.
    """
    contour_generator = contourpy.contour_generator(
        x=re_range,
        y=im_range,
        z=f( 1j*im_range.reshape(-1, 1) + re_range ),
        quad_as_tri=False,
        z_interp="Linear",
    )
    roots = [np.empty(shape=(0,), dtype=np.complex128)]
    for c in contour_generator.lines(0.0):
        re_contour = 1j * c[:, 1] + c[:, 0]
        crossings = _find_crossings(
            re_contour,
            np.sign( np.imag( f(re_contour) ) ),
            remove_consequent=True,
            interpolate=True,
        )
        if crossings.size:
            roots.append(crossings)
    return np.hstack(roots)


def spectrum_mapping(coefs: npt.NDArray, delays: npt.NDArray, rectangle: tuple[float, float, float, float], ds: float=None) -> npt.NDArray:
    """Locate quasi-polynomial roots in a rectangle via spectrum mapping.

    Parameters
    ----------
    coefs : ndarray
        Matrix of polynomial coefficients. Each row represents the coefficients
        corresponding to a specific delay.
    delays : ndarray
        Vector of delays associated with each row in ``coefs``.
    rectangle : tuple of float
        Search region ``(Re_min, Re_max, Im_min, Im_max)``.
    ds : float, optional
        Grid step. Required for the mapping grid.

    Returns
    -------
    roots : ndarray
        1D array of approximate complex roots.
    """
    roots = _spectrum_mapping(
        lambda s: quasipolynomial._eval_array(coefs, delays, s),
        re_range=np.arange(rectangle[0], rectangle[1], ds),
        im_range=np.arange(rectangle[2], rectangle[3], ds),
    )

    return roots
