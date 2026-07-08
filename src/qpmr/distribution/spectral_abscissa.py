r"""
Spectral abscissa bounds
========================

Functions for estimating bounds on the real parts of quasi-polynomial roots,
including delay-difference upper bounds and neutral vertical-strip limits.
"""

import logging

import numpy as np

from qpmr.quasipoly.core import compress, create_normalized_delay_difference_eq
from qpmr.core.quasipolynomial import compress, extract_delay_diff_eq
from qpmr.numerical_methods import newton

logger = logging.getLogger(__name__)


def _safe_upper_bound(ndiff_coefs, ndiff_delays, x0, tol, max_iter):
    """Solve for the unique positive root of a neutral bound equation.

    Parameters
    ----------
    ndiff_coefs : ndarray
        Coefficients of the normalized delay-difference equation.
    ndiff_delays : ndarray
        Delays associated with ``ndiff_coefs``.
    x0 : float
        Initial guess for Newton iteration.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum Newton iterations.

    Returns
    -------
    bound : float
        Estimated upper bound.
    converged : bool
        Whether Newton's method converged.
    """
    coefs_abs = np.abs(ndiff_coefs)
    bound, converged = newton(
        f=lambda x: np.inner(coefs_abs, np.exp(-x*ndiff_delays)) - 1.,
        f_prime=lambda x: np.inner(-ndiff_delays*coefs_abs, np.exp(-x*ndiff_delays)),
        x0=x0,
        tol=tol,
        max_iter=max_iter,
    )
    return bound


def _neutral_strip_bounds(diff_coefs, diff_delays, **kwargs) -> tuple[float, float]:
    """Return vertical-strip bounds for the neutral part of a quasi-polynomial.

    Parameters
    ----------
    diff_coefs : ndarray
        Coefficients of the delay-difference equation.
    diff_delays : ndarray
        Delays associated with ``diff_coefs``.

    Returns
    -------
    lb : float
        Lower bound on the real part of the neutral spectrum.
    ub : float
        Upper bound on the real part of the neutral spectrum.
    """
    ub = _safe_upper_bound(diff_coefs[1:]/diff_coefs[0], diff_delays[1:] - diff_delays[0], 0, 1e-6, 100)
    lb = -_safe_upper_bound(diff_coefs[:-1]/diff_coefs[-1], -diff_delays[:-1] + diff_delays[-1], 0, 1e-6, 100)
    return lb, ub


def safe_upper_bound_diff(coefs, delays, **kwargs):
    r"""Upper bound on the real part from the delay-difference equation.

    Computes a safe upper bound on the spectral abscissa using the normalized
    delay-difference representation of the quasi-polynomial.

    Parameters
    ----------
    coefs : ndarray
        Matrix of polynomial coefficients. Each row represents the coefficients
        corresponding to a specific delay.

    delays : ndarray
        Vector of delays associated with each row in ``coefs``.

    compress : bool, optional
        If ``True`` (default), compress ``coefs`` and ``delays`` before
        forming the delay-difference equation.

    Returns
    -------
    bound : float
        Upper bound estimate, or ``-inf`` if no delay-difference equation
        exists (e.g. retarded-only quasi-polynomial).
    """
    if kwargs.get("compress", True):
        coefs, delays = compress(coefs, delays)

    diff = create_normalized_delay_difference_eq(coefs, delays, compress=False)
    logger.debug(f"{diff}")
    if diff is None:
        return -np.inf

    diff_coefs, diff_delays = diff

    bound, converged = newton(
        f=lambda s: np.inner(diff_coefs, np.exp(-s*diff_delays)) - 1.,
        f_prime=lambda s: np.inner(-diff_delays*diff_coefs, np.exp(-s*diff_delays)),
        x0=0.,
        tol=1e-6,
        max_iter=100,
    )

    return bound


def bounds_neutral_strip(coefs, delays, **kwargs):
    """Bounds of the vertical strip associated with the neutral spectrum.

    Parameters
    ----------
    coefs : ndarray
        Matrix of polynomial coefficients. Each row represents the coefficients
        corresponding to a specific delay.

    delays : ndarray
        Vector of delays associated with each row in ``coefs``.

    compress : bool, optional
        If ``True`` (default), compress ``coefs`` and ``delays`` first.

    Returns
    -------
    ub : float
        Upper bound of the neutral vertical strip.
    lb : float
        Lower bound of the neutral vertical strip. Returns ``(inf, -inf)`` when
        no neutral part is present.
    """
    if kwargs.get("compress", True):
        coefs, delays = compress(coefs, delays)

    diff_coefs, diff_delays = extract_delay_diff_eq(coefs, delays, normalize=True, compress=False)
    if len(diff_coefs) <= 1:
        return np.inf, -np.inf

    ub, lb = _neutral_strip_bounds(diff_coefs, diff_delays)
    return ub, lb
