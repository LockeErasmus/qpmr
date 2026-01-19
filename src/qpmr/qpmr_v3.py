"""
QPmR v2 implementation
----------------------
Set of funtions implement original QPmR v2 algorithm, based on [1].

[1] Vyhlidal, Tomas, and Pavel Zitek. "Mapping based algorithm for large-scale
    computation of quasi-polynomial zeros." IEEE Transactions on Automatic
    Control 54.1 (2009): 171-177.


TODO:
    1. @overload docstring
"""
import logging
from typing import overload

import contourpy
import numpy as np
import numpy.typing as npt

from .numerical_methods import numerical_newton, secant, newton, mueller
from .argument_principle import argument_principle, argument_principle_circle, argument_principle_rectangle
from .zero_multiplicity import cluster_roots
from .common import find_crossings

from . import quasipoly

from .quasipoly import QuasiPolynomial
from .quasipoly.core import _eval_array
from .quasipoly.operation import derivative
# from .quasipoly.core import _eval_array_opt as _eval_array
from .grid import grid_size_heuristic
from .qpmr_metadata import QpmrInfo, QpmrSubInfo, QpmrRecursionContext
from .qpmr_validation import validate_region, validate_qp
from .region_heuristic import region_heuristic

logger = logging.getLogger(__name__)

IMPLEMENTED_NUMERICAL_METHODS = ["newton", "secant"]


def _qpmr(ctx: QpmrRecursionContext, region: tuple[float, float, float, float], e: float, ds: float, recursion_level: int, **kwargs):
    """ TODO
    
    Args:

        recursion_level (int): current splitting recursion level
    
    
    """
    # solve recursion
    if recursion_level > ctx.recursion_level_max:
        # TODO custom ERROR? : recursion error hit
        # TODO message for advices
        raise ValueError(f"Maximum level of allowed splitting recursion hit!")

    # create node object: TODO - this will need to be more nice
    if ctx.solution_tree is None:
        ctx.solution_tree = QpmrSubInfo()
        ctx.node = ctx.solution_tree
    else:
        ctx.node = QpmrSubInfo(parent=ctx.node)
    
    ctx.node.region = region
    ctx.node.ds = ds

    # TODO: if ds < machine precission

    # first, check if a little bit bigger region has roots
    perturbed_region = (region[0]-ds, region[1]+ds, region[2]-ds,region[3]+ds) # region expanded by ds
    
    n_argp = argument_principle_rectangle(ctx.f, perturbed_region, ds/10., eps=e/100., f_prime=ctx.f_prime)
    # n_argp = argument_principle(ctx.f, perturbed_region, ds/10., eps=e/100., f_prime=ctx.f_prime)
    if n_argp == 0:
        ctx.node.status = "SOLVED"
        ctx.node.status_message = "ARGP=0"
        ctx.node.roots = np.array([], dtype=np.complex128)
        logger.debug(f"SUCCESFULLY FINISHED BRANCH (ARGP=0)")
        return 

    # use region expanded by 3ds to each direction
    bmin, bmax, wmin, wmax = ctx.node.expanded_region

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
        correct_node = ctx.node
        for subregion in subregions:
            ctx.node.status = "FAILED"
            ctx.node.status_message = "GRID"
            _qpmr(ctx, subregion, e_new, ds_new, recursion_level+1)
            ctx.node = correct_node # re-set correct parent node before next iteration
        return
    
    # construct grid, add to metadata - grid is cached
    real_range = ctx.node.real_range
    imag_range = ctx.node.imag_range
    complex_grid = ctx.node.complex_grid # 1j*imag_range.reshape(-1, 1) + real_range
    func_value = ctx.f(complex_grid) # evaluates QP at grid points
    ctx.node.z_value = func_value

    ## finding contours via contourpy library, only 0-level real contours are necessary
    contour_generator = contourpy.contour_generator(x=real_range, y=imag_range, z=func_value.real)
    zero_level_contours = contour_generator.lines(0.0) # find all 0-level real contours
    ctx.node.contours_real = zero_level_contours

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

    ctx.node.roots0 = roots0

    if ctx.multiplicity_heuristic:
        roots0, roots_multiplicity = cluster_roots(roots0, eps=2*ds)
        logger.debug(f"{roots0}\n{roots_multiplicity}")
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
        ctx.node.roots_numerical = roots
        
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
    if n_expected == n_argp: # check if number matches expected number
        logger.debug(f"Argument principle success {n_argp}, expected: {n_expected} for region={perturbed_region}")
        argp_ok = True
    else:
        # try little  bit pertubed contour
        perturbed_region = (region[0]-0.9*ds, region[1]+0.9*ds, region[2]-0.9*ds,region[3]+0.9*ds)
        n_argp = argument_principle_rectangle(ctx.f, perturbed_region, ds/10., eps=e/100., f_prime=ctx.f_prime)
        if n_expected == n_argp: # check if number matches expected number
            logger.debug(f"Argument principle success {n_argp}, expected: {n_expected} for perturbed region={perturbed_region}")
            argp_ok = True

    if not argp_ok:
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
    ctx.node.status_message = "QPmR"
    ctx.node.roots = roots
    logger.debug(f"SUCCESFULLY FINISHED BRANCH (QPmR)")

@overload
def qpmr(
    coefs: npt.NDArray,
    delays: npt.NDArray,
    region: tuple[float, float, float, float]=None,
    **kwargs
) -> tuple[npt.NDArray[np.complex128], QpmrInfo]: ...

@overload
def qpmr(
    qp: QuasiPolynomial,
    region: tuple[float, float, float, float]=None,
    **kwargs
) -> tuple[npt.NDArray[np.complex128], QpmrInfo]: ...

def qpmr(*args, **kwargs) -> tuple[npt.NDArray[np.complex128], QpmrRecursionContext]:
    """ Quasi-polynomial Root Finder V3

    Attempts to find all roots of quasi-polynomial in rectangular subregion of
    complex plane using Quasi-polynomial Root Finder [1].

    Parameters
    ----------
    coefs : ndarray
        Matrix of polynomial coefficients. Each row represents the coefficients
        corresponding to a specific delay.

    delays : ndarray
        Vector of delays associated with each row in `coefs`.
    
    region : tuple of float, optional
        Definition of the rectangular region in the complex plane, specified as
        [Re_min, Re_max, Im_min, Im_max]. Defaults to None, which selects the
        region such that QPmR caputres 50 rightmost roots.

    e : float, optional
        Computation accuracy. Defaults to 1e-6.

    ds : float, optional
        Grid step. If not provided, a heuristic will be used to determine the step size.

    numerical_method : {'newton', 'secant'}, optional
        Numerical method used to refine the roots. Defaults to 'newton'.

    numerical_method_kwargs : dict, optional
        Additional keyword arguments passed to the numerical method. Defaults to None.

    grid_nbytes_max : int or None, optional
        Maximum allowed grid size in bytes. Defaults to 250e6. Set to None to disable
        the size check.

    Returns
    -------
    roots : ndarray
        Matrix of polynomial coefficients. Each row corresponds to a delay value.

    ctx : QpmrRecursionContext
        Tree-like object containing the computation metadata

        Attributes
        ----------
        TODO
    
    Notes
    -----

    .. math::

        h(s) = \sum_{i=0}^n p_i(s)e^{-s\tau_i}

    TODO

    References
    ----------
    .. [1] Vyhlidal, Tomas, and Pavel Zitek. "Mapping based algorithm for
           large-scale computation of quasi-polynomial zeros." IEEE 
           Transactions on Automatic Control 54.1 (2009): 171-177.

    Examples
    --------
    
    Example 1 from [1], i.e. quasi-polynomial :math: h(s) = s + e^{-s}``

    >>> import numpy as np
    >>> import qpmr
    >>> coefs = np.array([[0, 1],[1, 0.]])
    >>> delays = np.array([0, 1.])
    >>> roots, ctx = qpmr.qpmr(coefs, delays, region=(-10, 2, 0, 30))

    Visualize roots:

    >>> import matplotlib.pyplot as plt
    >>> import qpmr.plot
    >>> qpmr.plot.roots(roots)
    >>> plt.show()
    """
    # solve overload, unpack *args
    if len(args) == 2:
        qp, region = args
        coefs, delays = qp.coefs, qp.delays
    else: # len(args) == 3
        coefs, delays, region = args

    # Validate quasi-polynomial definition, TODO: consider to not run validation when class is passed as qp
    coefs, delays = validate_qp(coefs, delays)
    if region is None: # try to propose region by heuristic
        region = region_heuristic(coefs, delays, n=100)
        logger.debug(f"Region proposed via heuristic: {region=}")
    else:
        region = validate_region(region) # validates region and converts to tuple[float, float, float, float]
    
    # Warn if region defined in un-economical way (symetry by real axis)
    if region[2] < 0 and region[3] > 0:
        im_max = max(abs(region[2]), abs(region[3])) # better im_max
        logger.warning(
            (f"Spectra of quasi-polynomials with real coefficients are "
             f"symetrical by real axis, specified region {region=} is "
             f"unnecessarily large. It is advised to switch to "
             f"region=[{region[0]}, {region[1]}, 0, {im_max}]")
        )

    # transform to better form, such that spectrum of transformed
    # quasi-polynomial (+ possibly origin s=0) is equivalent to spectrum of
    # original quasi-polynomial TODO TODO TODO

    ccoefs, ddelays = quasipoly.compress(coefs, delays)
    # ccoefs, ddelays, spower = quasipoly.factorize_power(ccoefs, ddelays)
    # ccoefs, ddelays, tau_max = quasipoly.normalize_exponent(ccoefs, ddelays)
    tau_max = 1.0 # TODO
    # TODO what if representation trivial -> trivial result

    # adjust region
    if tau_max != 0:
        rregion = [z/tau_max for z in region]
    
    # Solve keyword arguments defaults
    e = kwargs.pop("e", 1e-6)
    ds = kwargs.pop("ds", None)
    if not ds:
        ds = grid_size_heuristic(rregion, ccoefs, ddelays)
        logger.debug(f"Grid size not specified, setting as ds={ds} (solved by heuristic)")
    else:
        ds = ds / tau_max if tau_max !=0 else ds

    numerical_method = kwargs.get("numerical_method", "newton")

    # kwargs check
    assert isinstance(e, float) and e > 0.0, "error 'e' numerical accuracy"
    assert isinstance(ds, float) and ds > 0.0, "error 'ds' grid stepsize"
    if numerical_method and numerical_method not in IMPLEMENTED_NUMERICAL_METHODS:
        raise ValueError(f"numerical_method='{numerical_method}' not implemented, available methods: {IMPLEMENTED_NUMERICAL_METHODS}")
    
    # create context - TODO
    ctx = QpmrRecursionContext(ccoefs, ddelays)
    ctx.ds = ds
    ctx.grid_nbytes_max = kwargs.get("grid_nbytes_max", 128_000_000)
    ctx.multiplicity_heuristic = kwargs.get("multiplicity_heuristic", False)
    ctx.numerical_method = kwargs.get("numerical_method", "newton")
    ctx.numerical_method_kwargs = kwargs.get("numerical_method_kwargs", dict())
    ctx.recursion_level_max = kwargs.get("recursion_level_max", 5)
    
    # validate context - TODO

    # run recursive QPmR algorithm
    _qpmr(ctx, rregion, e, ds, recursion_level=0)

    logger.debug(f"QPmR recursive solution tree:\n {ctx.render_tree}")
    return ctx.roots, ctx # TODO
