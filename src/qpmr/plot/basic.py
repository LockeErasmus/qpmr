"""
Basic root plots
----------------
"""

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import numpy.typing as npt

from qpmr import QpmrInfo

def _roots_matlab(roots: npt.NDArray, ax: Axes=None, **kwargs):
    """ Plots roots in the original QPmR V2 matlab style """
    if ax is None:
        _, ax = plt.subplots()
    
    if roots is None:
        roots = np.array([], dtype=np.complex128)
    
    ax.axhline(0.0, linestyle="-.", linewidth=0.5, color="k")
    ax.axvline(0.0, linestyle="-.", linewidth=0.5, color="k")


    ax.scatter(np.real(roots),
               np.imag(roots),
               marker="o",
               color="k",
               s=8,
               linewidths=0.5,
    )
    
    ax.set_xlabel(r"$\Re (\lambda)$")
    ax.set_ylabel(r"$\Im (\lambda)$")

    return ax

# def _roots_tds()

def roots(roots: npt.NDArray, ax: Axes=None, **kwargs):
    """Plot roots in the complex plane.

    Stable roots (negative real part) are green, unstable red, near imaginary
    axis blue. Style ``'matlab'`` uses filled circles.

    Parameters
    ----------
    roots : ndarray
        Complex roots to plot.
    ax : Axes, optional
        Matplotlib axes. Created if ``None``.
    tol : float, optional
        Tolerance for classifying real part as zero (default 1e-10).
    style : str, optional
        Plot style, ``'matlab'`` or default cross markers.

    Returns
    -------
    ax : Axes
        Matplotlib axes with the plot.
    """

    tol = kwargs.get("tol", 1e-10)

    if ax is None:
        _, ax = plt.subplots()
    
    if roots is None:
        roots = np.array([], dtype=np.complex128)
    
    ax.axhline(0.0, linestyle="-.", linewidth=1, color="k")
    ax.axvline(0.0, linestyle="-.", linewidth=1, color="k")

    style = kwargs.get("style", "matlab")
    match style:
        case "matlab":
            ax = _roots_matlab(roots, ax=ax, **kwargs)
            return ax
        case _:
                    

            roots_real = np.real(roots)
            roots_imag = np.imag(roots)

            mask_negative = roots_real < -tol
            if np.any(mask_negative) > 0:
                ax.scatter(roots_real[mask_negative],
                        roots_imag[mask_negative],
                        marker="x",
                        color="g",
                        linewidths=0.5,
                )
            
            mask_positive = roots_real > tol
            if np.any(mask_positive) > 0:
                ax.scatter(roots_real[mask_positive],
                        roots_imag[mask_positive],
                        marker="x",
                        color="r",
                        linewidths=0.5,
                )

            mask_zero = ~(mask_positive | mask_negative)
            if np.any(mask_zero) > 0:
                ax.scatter(roots_real[mask_zero],
                        roots_imag[mask_zero],
                        marker="x",
                        color="b",
                        linewidths=0.5,
                )
            
            ax.set_xlabel(r"$\Re (\lambda)$")
            ax.set_ylabel(r"$\Im (\lambda)$")

            return ax

def pole_zero(poles: npt.NDArray, zeros: npt.NDArray, ax: Axes=None, **kwargs):
    """Plot poles and zeros on one complex-plane axes.

    Parameters
    ----------
    poles : ndarray
        Complex poles (red ``x`` markers).
    zeros : ndarray
        Complex zeros (blue ``o`` markers).
    ax : Axes, optional
        Matplotlib axes. Created if ``None``.

    Returns
    -------
    ax : Axes
        Matplotlib axes with the plot.
    """
    if ax is None:
        _, ax = plt.subplots()
    
    if poles is None:
        poles = np.array([], dtype=np.complex128)
    if zeros is None:
        zeros = np.array([], dtype=np.complex128)

    ax.axhline(0.0, linestyle="-.", linewidth=1, color="k")
    ax.axvline(0.0, linestyle="-.", linewidth=1, color="k")
    ax.scatter(zeros.real, zeros.imag, marker="o", linewidths=0.5, color="b", s=10)
    ax.scatter(poles.real, poles.imag, marker="x", color="r", linewidths=0.5)
    ax.set_xlabel(r"$\Re (\lambda)$")
    ax.set_ylabel(r"$\Im (\lambda)$")
    return ax
    
def qpmr_contour(roots: npt.NDArray, meta:QpmrInfo=None, ax: Axes=None, **kwargs):
    """Plot roots with real and imaginary zero-level contours from QPmR metadata.

    Parameters
    ----------
    roots : ndarray
        Complex roots.
    meta : QpmrInfo, optional
        Metadata from :func:`qpmr.qpmr` containing grid and contour data.
    ax : Axes, optional
        Matplotlib axes. Created if ``None``.

    Returns
    -------
    ax : Axes
        Matplotlib axes with the plot.
    """
    if ax is None:
        _, ax = plt.subplots()
    
    ax.axhline(0.0, linestyle="-.", linewidth=1, color="k")
    ax.axvline(0.0, linestyle="-.", linewidth=1, color="k")

    # plot also contours stored in meta
    re_grid = np.real(meta.complex_grid)
    im_grid = np.imag(meta.complex_grid)
    ax.contour(re_grid, im_grid, np.real(meta.z_value), levels=[0],
                colors='blue', alpha=0.5)
    ax.contour(re_grid, im_grid, np.imag(meta.z_value), levels=[0],
                colors='green', alpha=0.5)
    
    roots_real = np.real(roots)
    roots_imag = np.imag(roots)

    ax.scatter(roots_real, roots_imag, marker="x", color="r", linewidths=0.5)
    
    ax.set_xlabel(r"$\Re (\lambda)$")
    ax.set_ylabel(r"$\Im (\lambda)$")
    
    return ax

def argument_principle_circle(roots: npt.NDArray, ds: float, ax: Axes=None):
    """Draw small circles around roots for multiplicity visualization.

    Parameters
    ----------
    roots : ndarray
        Complex roots.
    ds : float
        Grid step; circle radius is ``ds / 20``.
    ax : Axes, optional
        Matplotlib axes. Created if ``None``.

    Returns
    -------
    ax : Axes
        Matplotlib axes with circles added.
    """

    alpha=0.3
    color="red"

    if ax is None:
        _, ax = plt.subplots()

    for r in roots:
        circle = patches.Circle((r.real, r.imag), radius=ds/20., alpha=alpha, color=color)
        ax.add_patch(circle)
    
    return ax



