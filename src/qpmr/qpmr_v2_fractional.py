"""
Fractional QPmR v2 implementation
---------------------------------



Set of funtions implement original QPmR v2 algorithm, based on [1].

[1] Vyhlidal, Tomas, and Pavel Zitek. "Mapping based algorithm for large-scale
    computation of quasi-polynomial zeros." IEEE Transactions on Automatic
    Control 54.1 (2009): 171-177.
"""
from functools import cached_property
import logging
from typing import Callable

import contourpy
import numpy as np
import numpy.typing as npt

from .numerical_methods import numerical_newton, secant
from .argument_principle import argument_principle
from .qpmr_v2 import grid_size_heuristic, find_roots, QpmrOutputMetadata

logger = logging.getLogger(__name__)

IMPLEMENTED_NUMERICAL_METHODS = ["newton", "secant", None]


def eval_fractional_qp(z: npt.NDArray, coefs:npt.NDArray, delays:npt.NDArray, powers: npt.NDArray):
    """ TODO """
    val = np.sum(np.multiply(
        np.sum(
            (np.power(z[..., np.newaxis], powers[np.newaxis, :])[..., np.newaxis]
             * coefs.T[np.newaxis, ...]),
            axis=-2
        ),
        np.exp(z[..., np.newaxis] * - delays[np.newaxis, ...]),
    ), axis=-1)
    return val

def qpmr_fractional(
        region: list[float, float, float, float],
        coefs: npt.NDArray,
        delays: npt.NDArray,
        powers: npt.NDArray,
        **kwargs) -> tuple[npt.NDArray | None, QpmrOutputMetadata]:
    """ Fractional Quasi-polynomial Root Finder V2 TODO

    Attempts to find all roots of quasipolynomial in predefined region. See [1].

    [1] Vyhlidal, Tomas, and Pavel Zitek. "Mapping based algorithm for
    large-scale computation of quasi-polynomial zeros." IEEE Transactions on
    Automatic Control 54.1 (2009): 171-177.

    Args:
        region (list): definition of rectangular region in the complex plane of
            a form [Re_min, Re_max, Im_min, Im_max]
        coefs (array): matrix definition of polynomial coefficients (each row
            represents polynomial coefficients corresponding to delay)
        delays (array): vector definition of associated delays (each delay
            corresponds to row in `coefs`)

        **kwargs:
            e (float) - computation accuracy, default = 1e-6
            ds (float) - grid step, default obtained by heuristic
            numerical_method (str) - numerical method for increasing precission
                of roots, default "newton", other options: "secant"
            numerical_method_kwargs (dict) - keyword arguments for numerical
                method, default None
            grid_nbytes_max (int): maximal allowed size of grid in bytes,
                default 250e6 bytes, set to None to disregard maximum size check
                of the grid
    """
    # region check
    assert len(region) == 4, "region is expected to be of a form [Re_min, Re_max, Im_min, Im_max]"
    assert region[0] < region[1], f"region boundaries on real axis has to fullfill {region[0]} < {region[1]}"
    assert region[2] < region[3], f"region boundaries on imaginary axis has to fullfill {region[2]} < {region[3]}"

    # quasipolynomial definition check
    assert len(coefs.shape) == 2, "coefs have to be matrix (2D array)"
    assert len(delays.shape) == 1, "delays have to be vector (1D array)"
    assert coefs.shape[0] == delays.shape[0], "number of rows in coefs has to match number of delays"
    assert coefs.shape[1] > 0, "degree of quasipolynomial =/= 0"

    # advise if region defined in uneconomical way
    if region[2] < 0 and region[3] > 0:
        im_max = max(abs(region[2]), abs(region[3])) # better im_max
        logger.warning(
            (f"Spectra of quasi-polynomials are symetrical by real axis, "
             f"specified region {region=} is unnecessarily large. It is advised"
             f" to switch to region=[{region[0]}, {region[1]}, 0, {im_max}]")
        )
    
    # solve kwargs values
    e = kwargs.get("e", 1e-6)
    ds = kwargs.get("ds", None)
    if not ds:
        ds = grid_size_heuristic(region)
        logger.debug(f"Grid size not specified, setting as ds={ds} (solved by heuristic)")
    nbytes_max = kwargs.get("grid_nbytes_max", 250_000_000)
    numerical_method = kwargs.get("numerical_method", "newton")
    numerical_method_kwargs = kwargs.get("numerical_method_kwargs", dict())

    # kwargs check
    assert isinstance(e, float) and e > 0.0, "error 'e' numerical accuracy"
    assert isinstance(ds, float) and ds > 0.0, "error 'ds' grid stepsize"
    if numerical_method not in IMPLEMENTED_NUMERICAL_METHODS:
        raise ValueError(f"numerical_method='{numerical_method}' not implemented, available methods: {IMPLEMENTED_NUMERICAL_METHODS}")
    
    # create metadata object
    metadata = QpmrOutputMetadata()

    # extend region and create meshgrid (original algorithm)
    bmin=region[0] - 3*ds
    bmax=region[1] + 3*ds
    wmin=region[2] - 3*ds
    wmax=region[3] + 3*ds

    # estimate size of array in bytes np.complex128
    nbytes = ((bmax - bmin) // ds + 1) * ((wmax - wmin) // ds + 1) * 16 # 128 / 8 = bytes per complex number
    if nbytes_max is None:
        logger.warning("Disabled nbytes check - this may trigger swapping etc ...")
    elif nbytes > nbytes_max:
        raise ValueError((f"Estimated size of grid {nbytes} greater then {nbytes_max}. "
                          "Specify smaller region, or increase `grid_nbytes_max`."))
    else:
        logger.debug(f"Estimated size of complex grid = {nbytes} bytes")

    # construct grid, add to metadata - grid is cached
    real_range = np.arange(bmin, bmax, ds)
    imag_range = np.arange(wmin, wmax, ds)
    metadata.real_range = real_range
    metadata.imag_range = imag_range
    complex_grid = metadata.complex_grid # 1j*imag_range.reshape(-1, 1) + real_range
       
    # evaluate function, TODO: it is UNREADABLE!!!
    func_value = eval_fractional_qp(complex_grid, coefs, delays, powers)
    metadata.z_value = func_value
    func_value_real = np.real(func_value)
    # func_value_imag = np.imag(func_value) # not needed in original algorithm

    ## finding contours via contourpy library
    contour_generator = contourpy.contour_generator(x=real_range, y=imag_range, z=func_value_real)
    zero_level_contours = contour_generator.lines(0.0) # find all 0 level curves
    metadata.contours_real = zero_level_contours

    if not zero_level_contours: # list is is_empty, i.e []
        logger.warning(f"No real 0-level contours were found in region {region}.")
        return None, metadata
    
    # detecting intersection points 
    roots = []
    for polygon in zero_level_contours:
        polygon_complex = polygon[:,0] + 1j*polygon[:,1]
        polygon_func_value = eval_fractional_qp(polygon_complex, coefs, delays, powers) 
        # find all intersections
        polygon_func_imag = np.imag(polygon_func_value)
        crossings = find_roots(polygon_complex, polygon_func_imag)
        if crossings.size:
            roots.append(crossings)
    
    if not roots: # no crossings found
        logger.warning(f"No contour crossings found!") # TODO better message
        return None, metadata
    
    roots0 = np.hstack(roots)

    # apply numerical method to increase precission - TODO move to separate function `apply_numerical_method` ?
    func = lambda z: eval_fractional_qp(z, coefs, delays, powers)
    if numerical_method == "newton":
        roots, converged = numerical_newton(func, roots0, **numerical_method_kwargs)
    elif numerical_method == "secant":
        roots, converged = secant(func, roots0, **numerical_method_kwargs)
    
    if not converged: # if numerical method did not converge
        logger.info(f"'{numerical_method}' did not converged, ds <- ds / 3")
        modified_kwargs = kwargs.copy()
        modified_kwargs['ds'] = ds / 3.0
        return qpmr_fractional(region, coefs, delays, powers, **modified_kwargs)

    # filter out roots that are not in predefined region
    mask = ((roots.real >= region[0]) & (roots.real <= region[1]) # Re bounds
            & (roots.imag >= region[2]) & (roots.imag <= region[3])) # Im bounds
    roots = roots[mask]

    # Case where roots found, but are outside of defined region
    if roots.size == 0:
        logger.warning(f"No roots found in region!")
        return None, metadata
    
    # Check the distance from the approximation of the roots less then 2*ds
    dist = np.abs(roots - roots0[mask])
    num_dist_violations = (dist > 2*ds).sum()
    if num_dist_violations > 0:
        logger.info("After numerical method, MAX |roots0 - roots| > 2 * ds, ds <- ds / 3")
        modified_kwargs = kwargs.copy()
        modified_kwargs['ds'] = ds / 3.0
        return qpmr_fractional(region, coefs, delays, powers, **modified_kwargs)
    
    # Perform argument check
    n = argument_principle(func, region, ds/10., eps=e/100.) # ds and eps obtained from original matlab implementation
    smaller_region = [region[0]+ds/10., region[1]-ds/10., region[2]+ds/10.,region[3]-ds/10.]
    n_smaller = argument_principle(func, smaller_region, ds/10., eps=e/100.)
    if len(roots) == n or len(roots) == n_smaller:
        pass # ok, continue
    else:
        logger.info(f"Argument principle: {n}, real number of roots {len(roots)}")
        logger.info(f"Argument principle (smaller Region): {n_smaller}, real number of roots {len(roots)}")
        modified_kwargs = kwargs.copy()
        modified_kwargs['ds'] = ds / 3.0
        # return qpmr_fractional(region, coefs, delays, powers, **modified_kwargs)

    return roots, metadata
