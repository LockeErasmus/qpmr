"""
Argument Principle implementation
---------------------------------
TODO:
    1. implement different integration method
"""
import logging
from typing import Callable

import numpy as np
import numpy.typing as npt

from .quasipolynomial import _eval_array

logger = logging.getLogger(__name__)

def rectangular_contour(re_min, re_max, im_min, im_max) -> tuple[Callable, Callable, tuple[float, float]]:
    x0, y0, width, height = re_min, im_min, re_max-re_min, im_max-im_min
    
    def gamma(t):
        t = np.asarray(t)
        z = np.zeros_like(t, dtype=np.complex128)

        # Segment 1: bottom (left to right)
        mask1 = (0 <= t) & (t < 1)
        z[mask1] = x0 + (t[mask1] - 0) * width + 1j * y0

        # Segment 2: right (bottom to top)
        mask2 = (1 <= t) & (t < 2)
        z[mask2] = x0 + width + 1j * (y0 + (t[mask2] - 1) * height)

        # Segment 3: top (right to left)
        mask3 = (2 <= t) & (t < 3)
        z[mask3] = x0 + width - (t[mask3] - 2) * width + 1j * (y0 + height)

        # Segment 4: left (top to bottom)
        mask4 = (3 <= t) & (t <= 4)
        z[mask4] = x0 + 1j * (y0 + height - (t[mask4] - 3) * height)

        return z
    
    def gamma_prime(t):
        z = np.zeros_like(t, dtype=np.complex128)
        mask = (0 <= t) & (t < 1)
        z[mask] = width
        mask = (1 <= t) & (t < 2)
        z[mask] = 1j * height
        mask = (2 <= t) & (t < 3)
        z[mask] = -width
        mask = (3 <= t) & (t <= 4)
        z[mask] = -1j * height

        return z
    
    return gamma, gamma_prime, (0, 4)

def circle_contour(center: complex, radius: float):
    """Parametrize a circular contour in the complex plane.

    Parameters
    ----------
    center : complex
        Circle center.
    radius : float
        Circle radius.

    Returns
    -------
    gamma : callable
        Contour parametrization ``gamma(t)``.
    gamma_prime : callable
        Derivative ``gamma'(t)``.
    interval : tuple of float
        Parameter interval ``(0, 2*pi)``.
    """
    def gamma(t):
        return center + radius * np.exp(1j * t)
    def gamma_prime(t):
        return 1j * radius * np.exp(1j * t)
    return gamma, gamma_prime, (0, 2*np.pi)

def _discretize_rectangular_boundary(bmin: float, bmax: float, wmin: float, wmax: float, ds: float):
    """Discretize a rectangular boundary into a closed complex polyline.

    Parameters
    ----------
    bmin, bmax : float
        Real-axis bounds.
    wmin, wmax : float
        Imaginary-axis bounds.
    ds : float
        Approximate edge step size.

    Returns
    -------
    contour : ndarray
        Closed contour vertices (first equals last).
    contour_steps : ndarray
        Step vectors along each edge segment.
    """
    n_steps_real = int((bmax - bmin) / ds) + 1
    linspace_real = np.linspace(bmin, bmax, n_steps_real)
    step_real = (bmax - bmin) / (n_steps_real - 1)

    n_steps_imag = int((wmax - wmin) / ds) + 1
    linspace_imag = np.linspace(wmin - ds, wmax + ds, n_steps_imag)
    step_imag = (wmax - wmin + 2*ds) / (n_steps_imag - 1)

    contour = np.r_[linspace_real + 1j*wmin,
                    bmax + 1j*linspace_imag,
                    np.flip(linspace_real) + 1j*wmax,
                    bmin + 1j*np.flip(linspace_imag)]
    contour_steps = np.r_[np.full(shape=(n_steps_real,), fill_value=step_real),
                          np.full(shape=(n_steps_imag,), fill_value=1j*step_imag),
                          np.full(shape=(n_steps_real,), fill_value=-step_real),
                          np.full(shape=(n_steps_imag,), fill_value=-1j*step_imag)]
    return contour, contour_steps

def _argument_principle(f: Callable, f_prime: Callable, gamma: Callable, gamma_prime: Callable,  a: float, b: float, n_points: int=1000):
    """
    Compute (N - P) using the Argument Principle with vectorized integrand.

    Parameters:
    - f: function f(z)
    - df: derivative f'(z)
    - contour: function γ(t) that defines the contour in complex plane
    - a, b: parameter domain for the contour
    - n_points: number of discretization points

    Returns:
    - Approximation to number of zeros - number of poles inside contour
    """
    t = np.linspace(a, b, n_points)

    z = gamma(t)
    dz = gamma_prime(t)

    fz = f(z)
    dfz = f_prime(z)

    with np.errstate(divide='ignore', invalid='ignore'):
        integrand = np.where(fz != 0, dfz / fz * dz, 0.0)

    # TODO warning
    num_zeros = np.count_nonzero(fz == 0)
    if num_zeros > 0:
        print(f"Warning: encountered {num_zeros} zeros of f(z) on the contour.") # TODO

    integral = np.trapezoid(integrand, t)
    return integral / (2j * np.pi)

def _argument_principle_tracking(f: Callable, gamma: Callable, a: float, b: float, n_points: int=1000, zero_tol: float=1e-8):
    """Count zeros via unwrapped phase change along a contour.

    Parameters
    ----------
    f : callable
        Function evaluated on the contour.
    gamma : callable
        Contour parametrization.
    a, b : float
        Parameter interval endpoints.
    n_points : int, optional
        Number of discretization points.
    zero_tol : float, optional
        Tolerance for detecting zeros on the contour.

    Returns
    -------
    n : int
        Rounded net number of zeros minus poles inside the contour.
    """
    t = np.linspace(a, b, n_points)
    z = gamma(t)
    fz = f(z)

    mask = np.abs(fz) < zero_tol
    if np.any(mask):
        logger.warning("Function values too close to zero on contour for reliable argument principle evaluation.")
    
    theta = np.unwrap(np.angle(fz))
    n = int(np.round( (theta[-1] - theta[0]) / (2*np.pi)))
    return n

def _argument_principle_robust(f: Callable, gamma: Callable, a: float, b: float, n_points_0: int=256, n_max: int=1e12,
                               phase_tol: float=np.pi/2, zero_tol: float=1e-12):
    """Count zeros via adaptive phase tracking along a contour.

    Parameters
    ----------
    f : callable
        Function evaluated on the contour.
    gamma : callable
        Contour parametrization.
    a, b : float
        Parameter interval endpoints.
    n_points_0 : int, optional
        Initial discretization count.
    n_max : int, optional
        Maximum discretization count before failure.
    phase_tol : float, optional
        Maximum allowed phase increment between samples.
    zero_tol : float, optional
        Tolerance for zeros on the contour.

    Returns
    -------
    n : int or float
        Rounded zero count, or ``nan`` if refinement failed.
    """
    
    n_points = n_points_0
    success = False
    while n_points < n_max:
        t = np.linspace(a, b, n_points)
        z = gamma(t)
        fz = f(z)

        # check proximity to zero
        if np.any(np.abs(fz) < zero_tol):
            logger.error("Function values too close to zero on contour for reliable argument principle evaluation.")
            break

        theta = np.unwrap(np.angle(fz))
        dtheta = np.diff(theta)

        if np.all(np.abs(dtheta) < phase_tol):
            success = True
            break

        n_points *= 2  # double the number of points for finer discretization
        
    if not success:
        logger.error("Failed to achieve desired phase change tolerance within maximum number of points.")
        return np.nan
    
    n = int(np.round( (theta[-1] - theta[0]) / (2*np.pi)))
    return n
    

def argument_principle(f: Callable, region: tuple[float, float, float, float],
                       ds: float, eps: float) -> float:
    """Count roots in a rectangle via the argument principle.

    Parameters
    ----------
    f : callable
        Vectorized quasi-polynomial evaluator.
    region : tuple of float
        Rectangular region ``(Re_min, Re_max, Im_min, Im_max)``.
    ds : float
        Contour discretization step scale.
    eps : float
        Finite-difference step for the derivative.

    Returns
    -------
    n : float
        Rounded number of roots inside the region.
    """
    # prepare contour path
    re_min, re_max, im_min, im_max = region
    gamma, gamma_prime, (a,b) = rectangular_contour(re_min, re_max, im_min, im_max)

    def f_prime(s):
        dvals = (f(s - eps) - f(s + eps) + 1j*f(s +1j*eps)
                    - 1j*f(s -1j*eps)) / 4. / eps
        return dvals
    
    n_points = round(2*(re_max - re_min + im_max - im_min)/ds + 4)
    res = _argument_principle(f, f_prime, gamma, gamma_prime, a, b, n_points=n_points)

    # use argument principle and round
    n_raw = np.abs(np.real( res ))
    n = np.round(n_raw)
    logger.debug(f"Using argument principle, contour integral = {n_raw} | rounded to {n}")

    return n

def argument_principle_rectangle(f: Callable, region: tuple[float, float, float, float],
                                 ds: float, eps: float, f_prime=None) -> float:
    """Count roots in a rectangle via contour integration.

    Parameters
    ----------
    f : callable
        Vectorized quasi-polynomial evaluator.
    region : tuple of float
        Rectangular region ``(Re_min, Re_max, Im_min, Im_max)``.
    ds : float
        Contour discretization step scale.
    eps : float
        Finite-difference step when ``f_prime`` is not provided.
    f_prime : callable, optional
        Derivative of ``f``. Computed by central differences if ``None``.

    Returns
    -------
    n : float
        Rounded number of roots inside the region.
    """
    # prepare contour path
    re_min, re_max, im_min, im_max = region
    gamma, gamma_prime, (a,b) = rectangular_contour(re_min, re_max, im_min, im_max)

    if f_prime is None:
        f_prime = lambda s: ( f(s + eps) - f(s - eps)) / 2 / eps
        # def f_prime(s):
        #     dvals = (f(s - eps) - f(s + eps) + 1j*f(s +1j*eps)
        #                 - 1j*f(s -1j*eps)) / 4. / eps
        #     return dvals
    
    n_points = max(round(2*(re_max - re_min + im_max - im_min)/ds + 4), 1000)
    # res = _argument_principle(f, f_prime, gamma, gamma_prime, a, b, n_points=n_points)
    res = _argument_principle(f, f_prime, gamma, gamma_prime, a, b, n_points=n_points)

    # use argument principle and round
    n_raw = np.abs(np.real( res ))
    n = np.round(n_raw)
    logger.debug(f"Using argument principle, contour integral = {n_raw} | rounded to {n}")

    return n

def argument_principle_circle(f, circle: tuple[complex, float], ds: float, eps:float, f_prime: Callable=None):
    """Count roots inside a circle via the argument principle.

    Parameters
    ----------
    f : callable
        Vectorized quasi-polynomial evaluator.
    circle : tuple
        ``(center, radius)`` defining the circular contour.
    ds : float
        Angular step scale for discretization.
    eps : float
        Finite-difference step when ``f_prime`` is not provided.
    f_prime : callable, optional
        Derivative of ``f``.

    Returns
    -------
    n : float
        Rounded number of roots inside the circle.
    """
    # prepare contour path
    center, radius = circle
    gamma, gamma_prime, (a, b) = circle_contour(center, radius)

    if f_prime is None:
        f_prime = lambda s: ( f(s + eps) - f(s - eps)) / 2 / eps
    
    n_points = max(round(2*np.pi/ds) + 1, 1000)
    res = _argument_principle(f, f_prime, gamma, gamma_prime, a, b, n_points=n_points)

    # use argument principle and round
    n_raw = np.abs(np.real( res ))
    n = np.round(n_raw)
    logger.debug(f"Using argument principle, contour integral = {n_raw} | rounded to {n}")

    return n