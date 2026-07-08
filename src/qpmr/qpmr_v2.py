r"""
QPmR v2 implementation (legacy)
=================================

Original QPmR algorithm from Vyhlídal & Zítek (2009). Prefer :func:`qpmr.qpmr`
(v3) for new code.
"""
from functools import cached_property
import logging
from typing import overload

import contourpy
import numpy as np
import numpy.typing as npt

from .numerical_methods import numerical_newton, secant
from .core.argument_principle import argument_principle
from .core.mapping import _find_crossings as find_crossings
from .quasipoly import QuasiPolynomial
from .quasipoly.core import _eval_array
# from .quasipoly.core import _eval_array_opt as _eval_array
from .grid import grid_size_heuristic
from .qpmr_validation import validate_region, validate_qp

logger = logging.getLogger(__name__)

IMPLEMENTED_NUMERICAL_METHODS = ["newton", "secant"]

class QpmrInfoV2:
    """Computation metadata from legacy QPmR v2.

    Attributes
    ----------
    real_range, imag_range : ndarray
        Grid axes used for spectrum mapping.
    z_value : ndarray
        Quasi-polynomial values on the grid.
    roots0 : ndarray
        Initial root guesses from contour crossings.
    roots_numerical : ndarray
        Roots after numerical refinement.
    contours_real : list of ndarray
        Real zero-level contours from contourpy.
    """

    # TODO maybe dataclass? but solve cached property
    real_range: npt.NDArray = None
    imag_range: npt.NDArray = None
    z_value: npt.NDArray = None
    roots0: npt.NDArray = None
    roots_numerical: npt.NDArray = None

    contours_real: list[npt.NDArray] = None
    # contours_imag: list[npt.NDArray] = None

    @cached_property
    def complex_grid(self) -> npt.NDArray:
        return 1j*self.imag_range.reshape(-1, 1) + self.real_range
    
    @cached_property
    def contours_imag(self) -> list[npt.NDArray]:
        contour_generator = contourpy.contour_generator(
            x=self.real_range,
            y=self.imag_range,
            z=np.imag(self.z_value),
        )
        zero_level_contours = contour_generator.lines(0.0)
        return zero_level_contours

@overload
def qpmr(
    region: list[float, float, float, float],
    coefs: npt.NDArray,
    delays: npt.NDArray,
    **kwargs
) -> tuple[npt.NDArray[np.complex128], QpmrInfoV2]: ...

@overload
def qpmr(
    region: list[float, float, float, float],
    qp: QuasiPolynomial,
    **kwargs
) -> tuple[npt.NDArray[np.complex128], QpmrInfoV2]: ...

def qpmr(*args, **kwargs) -> tuple[npt.NDArray[np.complex128], QpmrInfoV2]:
    """Quasi-polynomial Root Finder V2 (legacy).

    Finds roots in a rectangular region using spectrum mapping and optional
    Newton or secant refinement. See Vyhlídal & Zítek (2009).

    Parameters
    ----------
    region : tuple of float
        Search region ``(Re_min, Re_max, Im_min, Im_max)``.
    coefs : ndarray
        Matrix of polynomial coefficients. Each row represents the coefficients
        corresponding to a specific delay.
    delays : ndarray
        Vector of delays associated with each row in ``coefs``.
    e : float, optional
        Target accuracy (default 1e-6).
    ds : float, optional
        Grid step; heuristic if omitted.
    numerical_method : {'newton', 'secant'}, optional
        Refinement method (default ``'newton'``).
    numerical_method_kwargs : dict, optional
        Extra arguments for the numerical method.
    grid_nbytes_max : int, optional
        Maximum grid size in bytes (default 250e6).

    Returns
    -------
    roots : ndarray
        Complex roots in ``region``.
    metadata : QpmrInfoV2
        Intermediate grids, contours, and guesses.

    Raises
    ------
    ValueError
        If validation fails or the grid exceeds ``grid_nbytes_max``.
    """
    # solve overload, unpack *args
    if len(args) == 2:
        region, qp = args
        coefs, delays = qp.coefs, qp.delays
    else: # len(args) == 3
        region, coefs, delays = args

    # Validate arguments
    region = validate_region(region) # validates region and converts to tuple[float, float, float, float]
    coefs, delays = validate_qp(coefs, delays)

    # Warn if region defined in un-economical way (symetry by real axis)
    if region[2] < 0 and region[3] > 0:
        im_max = max(abs(region[2]), abs(region[3])) # better im_max
        logger.warning(
            (f"Spectra of quasi-polynomials with real coefficients are "
             f"symetrical by real axis, specified region {region=} is "
             f"unnecessarily large. It is advised to switch to "
             f"region=[{region[0]}, {region[1]}, 0, {im_max}]")
        )
    
    # Solve keyword arguments defaults
    e = kwargs.get("e", 1e-6)
    ds = kwargs.get("ds", None)
    if not ds:
        ds = grid_size_heuristic(region, coefs, delays)
        logger.debug(f"Grid size not specified, setting as ds={ds} (solved by heuristic)")
    nbytes_max = kwargs.get("grid_nbytes_max", 250_000_000)
    numerical_method = kwargs.get("numerical_method", "newton")
    numerical_method_kwargs = kwargs.get("numerical_method_kwargs", dict())

    # kwargs check
    assert isinstance(e, float) and e > 0.0, "error 'e' numerical accuracy"
    assert isinstance(ds, float) and ds > 0.0, "error 'ds' grid stepsize"
    if numerical_method and numerical_method not in IMPLEMENTED_NUMERICAL_METHODS:
        raise ValueError(f"numerical_method='{numerical_method}' not implemented, available methods: {IMPLEMENTED_NUMERICAL_METHODS}")
    
    # create metadata object
    metadata = QpmrInfoV2()

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
        raise ValueError((f"Estimated size of grid {nbytes} greater then {nbytes_max}. Specify smaller `region` or "
                          f"increasing grid size `ds` is recommended. Alternatively, increase `grid_nbytes_max` or "
                          f"set it to None to turn off this safeguard completely."))
    else:
        logger.debug(f"Estimated size of complex grid = {nbytes} bytes")

    # construct grid, add to metadata - grid is cached
    real_range = np.arange(bmin, bmax, ds)
    imag_range = np.arange(wmin, wmax, ds)
    metadata.real_range = real_range
    metadata.imag_range = imag_range
    complex_grid = metadata.complex_grid # 1j*imag_range.reshape(-1, 1) + real_range
    func_value = _eval_array(coefs, delays, complex_grid) # evaluates QP at grid points
    metadata.z_value = func_value

    ## finding contours via contourpy library, only 0-level real contours are necessary
    contour_generator = contourpy.contour_generator(x=real_range, y=imag_range, z=func_value.real)
    zero_level_contours = contour_generator.lines(0.0) # find all 0-level real contours
    metadata.contours_real = zero_level_contours

    if not zero_level_contours: # no contours found -> no initial guesses
        logger.warning(f"No real 0-level contours were found in region {region}.")
        roots0 = np.array([], dtype=np.complex128)
    else: # detecting intersection points
        roots = [np.empty(shape=(0,), dtype=np.complex128)]
        logger.debug(f"Num. Re 0-level contours: {len(zero_level_contours)}")
        for polygon in zero_level_contours:
            polygon_complex = polygon[:,0] + 1j*polygon[:,1]
            polygon_func_value = _eval_array(coefs, delays, polygon_complex)
            # find all intersections
            polygon_func_imag = np.imag(polygon_func_value)
            crossings = find_crossings(polygon_complex, polygon_func_imag, interpolate=True)
            if crossings.size:
                roots.append(crossings)
        if not roots: # warn that no crossings found
            logger.warning(f"No contour crossings found!")
        roots0 = np.hstack(roots)

    metadata.roots0 = roots0

    if roots0.size > 0 and numerical_method:
        # apply numerical method to increase precission - TODO move to separate function `apply_numerical_method` ?
        func = lambda s: _eval_array(coefs, delays, s)
        if numerical_method == "newton":
            roots, converged = numerical_newton(func, roots0, **numerical_method_kwargs)
        elif numerical_method == "secant":
            roots, converged = secant(func, roots0, **numerical_method_kwargs)
        else:
            raise NotImplementedError(f"Numerical method '{numerical_method}' is not supported.")
        metadata.roots_numerical = roots
        
        if not converged: # if numerical method did not converge
            logger.info(f"'{numerical_method}' did not converge, ds <- ds / 3")
            modified_kwargs = kwargs.copy()
            modified_kwargs['ds'] = ds / 3.0
            return qpmr(region, coefs, delays, **modified_kwargs)
    
    else: # no numerical method applied
        roots = np.copy(roots0)

    if roots.size > 0:
        # round and filter out roots that are not in predefined region
        np.round(roots, decimals=10, out=roots) # TODO - questionable round decimals -> kwargs?
        mask = ((roots.real >= region[0]) & (roots.real <= region[1]) # Re bounds
                & (roots.imag >= region[2]) & (roots.imag <= region[3])) # Im bounds
        roots = roots[mask]

        if numerical_method:
            # Check the distance from the approximation of the roots less then 2*ds
            dist = np.abs(roots - roots0[mask])
            num_dist_violations = (dist > 2*ds).sum()
            if num_dist_violations > 0:
                logger.info("After numerical method, MAX |roots0 - roots| > 2 * ds, ds <- ds / 3")
                modified_kwargs = kwargs.copy()
                modified_kwargs['ds'] = ds / 3.0
                return qpmr(region, coefs, delays, **modified_kwargs)

    # Perform argument check, note regions adjusted as per in original implementation
    region1 = (region[0]-ds, region[1]+ds, region[2]-ds, region[3]+ds)
    region2 = (region1[0]+ds/10., region1[1]-ds/10., region1[2]+ds/10.,region1[3]-ds/10.)

    n1 = argument_principle(func, region1, ds/10., eps=e/100.)
    n2 = argument_principle(func, region2, ds/10., eps=e/100.)

    if len(roots) != n1 and len(roots) != n2:
        logger.info(f"Argument principle: {n1}, real number of roots {len(roots)}")
        logger.info(f"Argument principle (smaller Region): {n2}, real number of roots {len(roots)}")
        modified_kwargs = kwargs.copy()
        modified_kwargs['ds'] = ds / 3.0
        return qpmr(region, coefs, delays, **modified_kwargs)

    return roots, metadata
