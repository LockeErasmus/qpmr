r"""
QPmR v3 implementation
----------------------

New implementation of QPmR algorithm.

"""
from dataclasses import dataclass
import logging
from typing import Callable, overload
import collections

import numpy as np
import numpy.typing as npt

from .numerical_methods import numerical_newton, secant, newton, mueller
from .core.argument_principle import argument_principle_circle, argument_principle_rectangle
from .zero_multiplicity import cluster_roots
from .core.mapping import _spectrum_mapping

from . import quasipoly

from .quasipoly import QuasiPolynomial
from .quasipoly.core import _eval_array
from .quasipoly.operation import derivative
# from .quasipoly.core import _eval_array_opt as _eval_array
from .grid import grid_size_heuristic

from .qpmr_validation import validate_region, validate_qp
from .region_heuristic import region_heuristic

logger = logging.getLogger(__name__)

IMPLEMENTED_NUMERICAL_METHODS = ["newton", "secant"]

class QPmRNode:
    """Node in the QPmR recursive subdivision tree.

    Parameters
    ----------
    region : tuple of float
        Rectangular subregion as ``(Re_min, Re_max, Im_min, Im_max)``.
    ds : float
        Grid step size for spectrum mapping on this node.
    """

    def __init__(self, region: tuple[float, float, float, float], ds: float):
        self.region = region
        self.ds = ds

@dataclass
class QpmrInfo:
    """Metadata returned by :func:`qpmr`.

    Attributes
    ----------
    coefs : ndarray
        Compressed polynomial coefficients used during root finding.
    delays : ndarray
        Delays associated with each row in ``coefs``.
    region : tuple of float or None
        Search region ``(Re_min, Re_max, Im_min, Im_max)``.
    ds : float or None
        Grid step size used for spectrum mapping.
    solved_nodes : list of QPmRNode or None
        Nodes successfully solved (reserved for future use).
    unsolved_nodes : list of QPmRNode or None
        Nodes left unsolved (reserved for future use).
    """
    coefs: npt.NDArray[np.float64]
    delays: npt.NDArray[np.float64]

    region: tuple[float, float, float, float] = None
    ds: float = None

    solved_nodes: list[QPmRNode] = None
    unsolved_nodes: list[QPmRNode] = None


def _qpmr_solve_node(f: Callable, f_prime: Callable, region: tuple[float, float, float, float], e: float, ds: float, **kwargs):
    """Solve a single rectangular node of the QPmR subdivision tree.

    Applies argument principle, spectrum mapping, optional numerical refinement,
    and validation checks for one subregion.

    Parameters
    ----------
    f : callable
        Quasi-polynomial evaluated on a complex grid.
    f_prime : callable
        Derivative of ``f``.
    region : tuple of float
        Rectangular subregion ``(Re_min, Re_max, Im_min, Im_max)``.
    e : float
        Target accuracy.
    ds : float
        Grid step for spectrum mapping.
    **kwargs
        Passed through to numerical refinement (e.g. ``numerical_method``).

    Returns
    -------
    roots : ndarray or None
        Roots found in ``region``, or ``None`` if the node could not be solved.
    meta : dict
        Status message (e.g. ``{"message": "SOLVED"}``).
    """

    multiplicity_heuristic = kwargs.get("multiplicity_heuristic", False)
    numerical_method = kwargs.get("numerical_method", "newton")

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

def qpmr(*args, **kwargs) -> tuple[npt.NDArray[np.complex128], QpmrInfo]:
    r"""Quasi-polynomial Root Finder V3

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
        [Re_min, Re_max, Im_min, Im_max]. Defaults to None, which attempts to
        use region heuristic applicable to retarded and neutral
        quasi-polynomials such that it captures 100 most decisive roots.

    e : float, optional
        Computation accuracy. Defaults to 1e-6.

    ds : float, optional
        Grid step. If not provided, a heuristic will be used to determine the step size.

    numerical_method : {'newton', 'secant'}, optional
        Numerical method used to refine the roots. Defaults to 'newton'.

    numerical_method_kwargs : dict, optional
        Additional keyword arguments passed to the numerical method. Defaults to None.

    grid_nbytes_max : int or None, optional
        Maximum allowed grid size in bytes. Defaults to 32e6. Set to None to disable
        the size check.

    Returns
    -------
    roots : ndarray
        1D array of complex roots found inside the search region.

    ctx : QpmrInfo
        Metadata from the computation (compressed coefficients, delays, and
        region).

    Raises
    ------
    ValueError
        If arguments are invalid or keyword options are inconsistent.
    NotImplementedError
        If a callable is passed instead of coefficient arrays.

    Notes
    -----

    .. math::

        h(s) = \sum_{i=0}^n p_i(s)e^{-s\tau_i}

    References
    ----------
    .. [1] Vyhlidal, Tomas, and Pavel Zitek. "Mapping based algorithm for
           large-scale computation of quasi-polynomial zeros." IEEE
           Transactions on Automatic Control 54.1 (2009): 171-177.

    Examples
    --------

    Example 1 from [1], quasi-polynomial :math:`h(s) = s + e^{-s}`:

    >>> import numpy as np
    >>> import qpmr
    >>> coefs = np.array([[0, 1],[1, 0.]])
    >>> delays = np.array([0, 1.])
    >>> roots, info = qpmr.qpmr(coefs, delays, region=(-10, 2, 0, 30))

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
    # original quasi-polynomial - TODO?

    coefs, delays = quasipoly.compress(coefs, delays)
    
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
    
    return np.hstack(roots_solution), QpmrInfo(coefs=coefs, delays=delays, region=region)
