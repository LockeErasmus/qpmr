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
from typing import Callable, overload
import collections

import numpy as np
import numpy.typing as npt

from .numerical_methods import numerical_newton, secant, newton, mueller
from .core.argument_principle import argument_principle_circle, argument_principle_rectangle
from .zero_multiplicity import cluster_roots
from .core.spectrum_mapping import _spectrum_mapping

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

class QPmRNode:
    def __init__(self, region: tuple[float, float, float, float], ds: float):
        self.region = region
        self.ds = ds

def _qpmr_solve_node(f: Callable, f_prime: Callable, region: tuple[float, float, float, float], e: float, ds: float, **kwargs):
    """ TODO """

    multiplicity_heuristic = kwargs.get("multiplicity_heuristic", False)
    numerical_method = kwargs.get("numerical_method", None)

    # TODO: what if region is so big I cannot even evaluate argument principle on it?
    #  - this is currently handled by recursion level and grid size rules,
    # but maybe we can do something smarter here

    # TODO: what if ds too small? Default to Weil?

    # step 1: check bigger if region has roots using argument principle
    n_argp = argument_principle_rectangle(
        f,
        (region[0]-ds, region[1]+ds, region[2]-ds,region[3]+ds),
        ds/10.,
        eps=e/100.
    )
    if n_argp == 0:
        meta = {"message": "ARGP=0"}
        return np.array([], dtype=np.complex128), meta

    # step 2: if region has roots, apply spectrum mapping to find approximations of roots
    roots0 =_spectrum_mapping(
        f,
        np.arange(region[0]-3*ds, region[1]+3*ds, ds),
        np.arange(region[2]-3*ds, region[3]+3*ds, ds),
    )

    # (optional) step: apply clustering to detect multiple roots
    if multiplicity_heuristic:
        roots0, roots_multiplicity = cluster_roots(roots0, eps=2*ds)
        multiplicity_unique, multiplicity_counts = np.unique(roots_multiplicity, return_counts=True)
        logger.debug((f"Clustering roots heuristic (DBSCAN with eps={2*ds})\n"
                    f"    multiplicities: {multiplicity_unique}\n"
                    f"    counts        : {multiplicity_counts}"))
    else:
        roots_multiplicity = np.full_like(roots0, fill_value=1, dtype=np.int64)
    
    if roots0.size > 0 and numerical_method:
        # apply numerical method to increase precission - TODO move to separate function `apply_numerical_method` ?
        # func = lambda s: _eval_array(ctx.coefs, ctx.delays, s)
        if numerical_method == "newton":
            roots, converged = numerical_newton(f, roots0, **kwargs.get("numerical_method_kwargs", {}))
            # roots, converged = _newton_array(ctx.f, ctx.f_prime, roots0) # TODO kwargs
        elif numerical_method == "secant":
            roots, converged = secant(f, roots0, **kwargs.get("numerical_method_kwargs", {}))
        else:
            raise NotImplementedError(f"Numerical method '{numerical_method}' is not supported.")
        
        if not converged: # if numerical method did not converge
            # TODO status
            logger.info(f"'{numerical_method}' did not converge")
            meta = {"message": "NUM_EPS"}
            return None, meta    
    else: # no numerical method applied
        roots = np.copy(roots0)

    if roots.size > 0:
        # round and filter out roots that are not in predefined region
        # np.round(roots, decimals=10, out=roots) # TODO - questionable round decimals -> kwargs?
        mask = ((roots.real >= region[0]) & (roots.real <= region[1]) # Re bounds
                & (roots.imag >= region[2]) & (roots.imag <= region[3])) # Im bounds
        roots = roots[mask]
        roots_multiplicity = roots_multiplicity[mask]

        if numerical_method:
            # Check the distance from the approximation of the roots less then 2*ds
            dist = np.abs(roots - roots0[mask])
            num_dist_violations = (dist > 2*ds).sum()
            if num_dist_violations > 0:
                logger.info("After numerical method, MAX |roots0 - roots| > 2 * ds")
                meta = {"message": "NUM_DS"}
                return None, meta
    
    # Perform argument check, note regions adjusted as per in original implementation
    argp_ok = False
    n_expected = np.sum(roots_multiplicity)
    if n_expected == n_argp: # check if number matches expected number
        logger.debug(f"Argument principle success {n_argp}, expected: {n_expected}")
        argp_ok = True
    else:
        # try little  bit pertubed contour
        perturbed_region = (region[0]-0.9*ds, region[1]+0.9*ds, region[2]-0.9*ds,region[3]+0.9*ds)
        n_argp = argument_principle_rectangle(
            f,
            perturbed_region,
            ds/10.,
            eps=e/100.,
            f_prime=f_prime)
        if n_expected == n_argp: # check if number matches expected number
            logger.debug(f"Argument principle success {n_argp}, expected: {n_expected} for perturbed (smaller region)")
            argp_ok = True

    if not argp_ok:
        meta = {"message": "ARGP"}
        return None, meta
    
    # Perform argument check for all multiplicities > 1
    # dcoefs, ddelays = derivative(coefs, delays)
    if multiplicity_heuristic:
        for i in np.where(roots_multiplicity > 1)[0]:
            r, rm = roots[i], roots_multiplicity[i]
            n = argument_principle_circle(
                    f,
                    (r, ds/20.),
                    ds/800.,
                    eps=ds/8000.,
                    f_prime=f_prime,
                )
            if n != rm:
                meta = {"message": "MULT_HEURISTIC"}
                logger.debug(f"MULTIPLICITY HEURISTIC ERROR: root={r}| {n=} ({rm})")
                return None, meta
    
    return roots, {"message": "SOLVED"}

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
    
    roots0 = _spectrum_mapping(ctx.f, real_range, imag_range)
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
    **kwargs
) -> tuple[npt.NDArray[np.complex128], QpmrInfo]: ...

@overload
def qpmr(
    qp: QuasiPolynomial,
    **kwargs
) -> tuple[npt.NDArray[np.complex128], QpmrInfo]: ...

@overload
def qpmr(
    qp: Callable,
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
    if len(args) == 1 and isinstance(args[0], QuasiPolynomial):
        qp = args[0]
        coefs, delays = qp.coefs, qp.delays
    elif len(args) == 1 and callable(args[0]):
        f = args[0]
        # TODO region heuristic for callable, currently not implemented
        raise NotImplementedError("QPmR for callable is not implemented yet, please provide coefs and delays directly.")
    elif len(args) == 2 and isinstance(args[0], np.ndarray) and isinstance(args[1], np.ndarray):
        coefs, delays = validate_qp(args[0], args[1]) # validate coefs and delays
    else:
        raise ValueError("Invalid arguments, expected (QuasiPolynomial) or (ndarray, ndarray)")   

    region = kwargs.pop("region", None)
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

    coefs, delays = quasipoly.compress(coefs, delays)
    # ccoefs, ddelays, spower = quasipoly.factorize_power(ccoefs, ddelays)
    # ccoefs, ddelays, tau_max = quasipoly.normalize_exponent(ccoefs, ddelays)
    
    # Solve keyword arguments defaults
    ds = kwargs.pop("ds", None)
    if not ds:
        ds = grid_size_heuristic(region, coefs, delays)
        logger.debug(f"Grid size not specified, setting as ds={ds} (solved by heuristic)")
    else:
        if not isinstance(ds, (float, int)) or ds <= 0.0:
            raise ValueError("Invalid 'ds' argument, expected positive number.")
    
    e = kwargs.pop("e", 1e-6)
    if not isinstance(e, (float, int)) or e <= 0.0:
        raise ValueError("Invalid 'e' argument, expected positive number.")
    
    numerical_method = kwargs.get("numerical_method", "newton")
    if numerical_method and numerical_method not in IMPLEMENTED_NUMERICAL_METHODS:
        raise ValueError(f"numerical_method='{numerical_method}' not implemented, available methods: {IMPLEMENTED_NUMERICAL_METHODS}")
    
    # create context - TODO
    coefs_prime, delays_prime = derivative(coefs, delays)
    f = lambda s: _eval_array(coefs, delays, s)
    f_prime = lambda s: _eval_array(coefs_prime, delays_prime, s)

    
    queue = collections.deque([QPmRNode(region, ds)])
    grid_nbytes_max = kwargs.get("grid_nbytes_max", 32_000_000)
    if grid_nbytes_max is None:
        logger.warning("Disabled nbytes recursion rule - this may trigger memory swapping and") 

    roots_solution = []
    for i in range(100):
        logger.info(f"Recursion level: {i}, queue size: {len(queue)}, current ds: {queue[0].ds if queue else 'N/A'}")
        if not queue:
            break # solved succesfully

        node = queue.popleft()

        # 128 / 8 = bytes per complex number
        grid_nbytes = (((node.region[1] - node.region[0]) // node.ds + 1) 
                       * ((node.region[3] - node.region[2]) // node.ds + 1) * 16)
        logger.debug(f"Estimated grid size in bytes: {grid_nbytes}, max allowed: {grid_nbytes_max}")
        
        if grid_nbytes > grid_nbytes_max:
            queue.extend([
                QPmRNode((node.region[0], node.region[0] + 0.5*(node.region[1] - node.region[0]), node.region[2], node.region[2] + 0.5*(node.region[3] - node.region[2])), node.ds), # left-bottom
                QPmRNode((node.region[0], node.region[0] + 0.5*(node.region[1] - node.region[0]), node.region[2] + 0.5*(node.region[3] - node.region[2]) + e, node.region[3]), node.ds), # left-top
                QPmRNode((node.region[0] + 0.5*(node.region[1] - node.region[0]) + e, node.region[1], node.region[2], node.region[2] + 0.5*(node.region[3] - node.region[2])), node.ds), # right-bottom
                QPmRNode((node.region[0] + 0.5*(node.region[1] - node.region[0]) + e, node.region[1], node.region[2] + 0.5*(node.region[3] - node.region[2]) + e, node.region[3]), node.ds), # right-top
            ])
            continue

        roots, meta = _qpmr_solve_node(f, f_prime, node.region, e, node.ds, **kwargs)

        if roots is None:
            logger.debug(f"Failed to solve node with region={node.region}, ds={node.ds}, meta={meta}")
            # TODO handle different meta messages
            queue.append(QPmRNode(node.region, node.ds/2.))
        else:
            roots_solution.append(roots)
    
    if queue:
        logger.warning(f"QPmR finished with non-empty queue, some branches were not solved, remaining branches: {len(queue)}")
    
    return np.hstack(roots_solution), QpmrRecursionContext(coefs, delays)
