"""
QPmR v2 implementation
----------------------
Set of funtions implement original QPmR v2 algorithm, based on [1].

[1] Vyhlidal, Tomas, and Pavel Zitek. "Mapping based algorithm for large-scale
    computation of quasi-polynomial zeros." IEEE Transactions on Automatic
    Control 54.1 (2009): 171-177.
"""
import logging
from typing import overload

import contourpy
import numpy as np
import numpy.typing as npt

from .numerical_methods import numerical_newton, secant, newton
from .argument_principle import argument_principle, argument_principle_circle
from .zero_multiplicity import cluster_roots
from .common import find_crossings
from .quasipoly import QuasiPolynomial
from .quasipoly.core import _eval_array
from .quasipoly.operation import derivative
# from .quasipoly.core import _eval_array_opt as _eval_array
from .grid import grid_size_heuristic
from .qpmr_metadata import QpmrInfo, QpmrSubInfo, QpmrRecursionContext
from .qpmr_validation import validate_region, validate_qp

logger = logging.getLogger(__name__)

IMPLEMENTED_NUMERICAL_METHODS = ["newton", "secant"]


def _qpmr(ctx: QpmrRecursionContext, region: tuple[float, float, float, float], e: float, ds: float, recursion_level: int, **kwargs):
    """
    
    Args:

        recursion_level (int): current splitting recursion level
    
    
    """
    # solve recursion
    if recursion_level > ctx.recursion_level_max:
        # TODO custom ERROR? : recursion error hit
        # TODO message for advices
        raise ValueError(f"Maximum level of allowed splitting recursion hit!")

    # create metadata object: TODO - this will need to be more nice
    if ctx.solution_tree is None:
        ctx.solution_tree = QpmrSubInfo()
        ctx.node = ctx.solution_tree
        metadata = ctx.solution_tree
    else:
        ctx.node = QpmrSubInfo(parent=ctx.node)
        metadata = ctx.node
    
    metadata.region = region
    metadata.ds = ds

    # TODO: if ds < machine precission

    # solve grid
    bmin, bmax = region[0] - 3*ds, region[1] + 3*ds
    wmin, wmax = region[2] - 3*ds, region[3] + 3*ds

    # estimate size of array in bytes np.complex128
    grid_nbytes = ((bmax - bmin) // ds + 1) * ((wmax - wmin) // ds + 1) * 16 # 128 / 8 = bytes per complex number
    
    logger.debug(f"level={recursion_level} / {ctx.recursion_level_max} - {ds=}, {region=}")
    if ctx.grid_nbytes_max is None:
        logger.warning("Disabled nbytes recursion rule - this may trigger memory swapping and")    
    elif grid_nbytes > ctx.grid_nbytes_max:        
        # split region into 2x2 grid of subregions
        ds_new = ds
        e_new = e
        width, height = region[1] - region[0], region[3] - region[2]
        subregions = [
            (region[0], region[0] + 0.5*width, region[2], region[2] + 0.5*height), # left-bottom
            (region[0], region[0] + 0.5*width, region[2] + 0.5*height + e, region[3]), # left-top
            (region[0] + 0.5*width + e, region[1], region[2], region[2] + 0.5*height), # right-bottom
            (region[0] + 0.5*width + e, region[1], region[2] + 0.5*height + e, region[3]), # right-top
        ]
        for subregion in subregions:
            metadata.status = "FAILED"
            ctx.node.status_message = "GRID"
            ctx.node = metadata # re-set correct parent node
            _qpmr(ctx, subregion, e_new, ds_new, recursion_level+1)
        
        return
    
    # construct grid, add to metadata - grid is cached
    real_range = np.arange(bmin, bmax, ds)
    imag_range = np.arange(wmin, wmax, ds)
    metadata.real_range = real_range
    metadata.imag_range = imag_range
    complex_grid = metadata.complex_grid # 1j*imag_range.reshape(-1, 1) + real_range
    func_value = ctx.f(complex_grid) # evaluates QP at grid points
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
            # find all intersections
            polygon_func_imag = np.imag(ctx.f(polygon_complex))
            crossings = find_crossings(polygon_complex, polygon_func_imag, interpolate=True)
            if crossings.size:
                roots.append(crossings)
        if not roots: # warn that no crossings found
            logger.warning(f"No contour crossings found!")
        roots0 = np.hstack(roots)

    metadata.roots0 = roots0

    if ctx.multiplicity_heuristic:
        roots0, roots_multiplicity = cluster_roots(roots0, eps=2*ds)
        multiplicity_unique, multiplicity_counts = np.unique(roots_multiplicity, return_counts=True)
        logger.debug((f"Clustering roots heuristic (DBSCAN with eps={2*ds})\n"
                    f"    multiplicities: {multiplicity_unique}\n"
                    f"    counts        : {multiplicity_counts}"))
    else:
        roots_multiplicity = np.full_like(roots0, fill_value=1, dtype=np.int64)
    
    if roots0.size > 0 and ctx.numerical_method:
        # apply numerical method to increase precission - TODO move to separate function `apply_numerical_method` ?
        # func = lambda s: _eval_array(ctx.coefs, ctx.delays, s)
        if ctx.numerical_method == "newton":
            roots, converged = numerical_newton(ctx.f, roots0, **ctx.numerical_method_kwargs)
            # roots, converged = _newton_array(ctx.f, ctx.f_prime, roots0) # TODO kwargs
        elif ctx.numerical_method == "secant":
            roots, converged = secant(ctx.f, roots0, **ctx.numerical_method_kwargs)
        else:
            raise NotImplementedError(f"Numerical method '{ctx.numerical_method}' is not supported.")
        metadata.roots_numerical = roots
        
        if not converged: # if numerical method did not converge
            # TODO status
            logger.info(f"'{ctx.numerical_method}' did not converge, ds <- ds / 3")
            ctx.node.status = "FAILED"
            ctx.node.status_message = "NUM_EPS"
            _qpmr(ctx, region, e, ds/2, recursion_level)
            return
    
    else: # no numerical method applied
        roots = np.copy(roots0)

    if roots.size > 0:
        # round and filter out roots that are not in predefined region
        # np.round(roots, decimals=10, out=roots) # TODO - questionable round decimals -> kwargs?
        mask = ((roots.real >= region[0]) & (roots.real <= region[1]) # Re bounds
                & (roots.imag >= region[2]) & (roots.imag <= region[3])) # Im bounds
        roots = roots[mask]
        roots_multiplicity = roots_multiplicity[mask]

        if ctx.numerical_method:
            # Check the distance from the approximation of the roots less then 2*ds
            dist = np.abs(roots - roots0[mask])
            num_dist_violations = (dist > 2*ds).sum()
            if num_dist_violations > 0:
                logger.info("After numerical method, MAX |roots0 - roots| > 2 * ds, ds <- ds / 3")
                ctx.node.status = "FAILED"
                ctx.node.status_message = "NUM_DS"
                _qpmr(ctx, region, e, ds/2, recursion_level)
                return
    
    # Perform argument check, note regions adjusted as per in original implementation
    argp_ok = False
    n_expected = np.sum(roots_multiplicity)
    regions_to_check = [
        region,
        (region[0]+ds/10., region[1]-ds/10., region[2]+ds/10.,region[3]-ds/10.), # smaller region
    ]
    for region_to_check in regions_to_check:
        n_argp = argument_principle(
            ctx.f,
            region_to_check,
            ds/10.,
            eps=e/100.
        )
        if n_argp == n_expected:
            argp_ok = True
            break
        logger.debug(f"Argument principle failed: {n_argp}, expected: {n_expected}")
    
    if argp_ok:
        logger.debug(f"Argument principle success {n_argp}, expected: {n_expected} for region={region}")
    else:
        ctx.node.status = "FAILED"
        ctx.node.status_message = "ARGP"
        _qpmr(ctx, region, e, ds/2, recursion_level)
        return
    
    # Perform argument check for all multiplicities > 1
    # dcoefs, ddelays = derivative(coefs, delays)
    if ctx.multiplicity_heuristic:
        for i in np.where(roots_multiplicity > 1)[0]:
            r, rm = roots[i], roots_multiplicity[i]
            n = argument_principle_circle(
                    ctx.f,
                    (r, ds/20.),
                    ds/800.,
                    eps=ds/8000.,
                    f_prime=ctx.f_prime,
                )
            if n != rm:
                logger.debug(f"MULTIPLICITY HEURISTIC ERROR: root={r}| {n=} ({rm})")
                ctx.node.status = "FAILED"
                ctx.node.status_message = "MULT_HEURISTIC"
                _qpmr(ctx, region, e, ds/2, recursion_level)
                return

    # SOLVED: add solution to tree TODO
    ctx.node.status = "SOLVED"
    ctx.node.status_message = None
    ctx.node.roots = roots
    logger.debug(f"SUCCESFULLY FINISHED BRANCH")

@overload
def qpmr(
    region: list[float, float, float, float],
    coefs: npt.NDArray,
    delays: npt.NDArray,
    **kwargs
) -> tuple[npt.NDArray[np.complex128], QpmrInfo]: ...

@overload
def qpmr(
    region: list[float, float, float, float],
    qp: QuasiPolynomial,
    **kwargs
) -> tuple[npt.NDArray[np.complex128], QpmrInfo]: ...

def qpmr(*args, **kwargs) -> tuple[npt.NDArray[np.complex128], QpmrRecursionContext]:
    """ Quasi-polynomial Root Finder V2

    Attempts to find all roots of quasi-polynomial in rectangular subregion of
    complex plane. For more details, see:

    [1] Vyhlidal, Tomas, and Pavel Zitek. "Mapping based algorithm for
    large-scale computation of quasi-polynomial zeros." IEEE Transactions on
    Automatic Control 54.1 (2009): 171-177.

    TODO Overload:
        ...

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
    e = kwargs.pop("e", 1e-6)
    ds = kwargs.pop("ds", None)
    if not ds:
        ds = grid_size_heuristic(region, coefs, delays)
        logger.debug(f"Grid size not specified, setting as ds={ds} (solved by heuristic)")

    numerical_method = kwargs.get("numerical_method", "newton")

    # kwargs check
    assert isinstance(e, float) and e > 0.0, "error 'e' numerical accuracy"
    assert isinstance(ds, float) and ds > 0.0, "error 'ds' grid stepsize"
    if numerical_method and numerical_method not in IMPLEMENTED_NUMERICAL_METHODS:
        raise ValueError(f"numerical_method='{numerical_method}' not implemented, available methods: {IMPLEMENTED_NUMERICAL_METHODS}")
    

    # create context - TODO
    ctx = QpmrRecursionContext(coefs, delays)
    ctx.ds = ds
    ctx.grid_nbytes_max = kwargs.get("grid_nbytes_max", 128_000_000)
    ctx.multiplicity_heuristic = kwargs.get("multiplicity_heuristic", False)
    ctx.numerical_method = kwargs.get("numerical_method", "newton")
    ctx.numerical_method_kwargs = kwargs.get("numerical_method_kwargs", dict())

    # validate context - TODO





    # run recursive QPmR algorithm
    _qpmr(ctx, region, e, ds, recursion_level=0)

    return ctx.roots, ctx # TODO
