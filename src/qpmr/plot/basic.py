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

def roots(roots: npt.NDArray, ax: Axes=None, **kwargs):
    """ Plots roots ('x') into complex plane

    Roots have different color stable - green, margin of stability - blue
    unstable - red
    
    Args:
        roots (array): array of complex numbers, if None treated as empty array
        ax (Axes): matplotlib Axes object, if None new figure, ax is created
        **kwargs:
            tol (float): tolerance for assuming Re(root) ~ 0, defgault 1e-10

    Returns:
        ax (Axes): matplotlib Axes object
    """

    tol = kwargs.get("tol", 1e-10)

    if ax is None:
        _, ax = plt.subplots()
    
    if roots is None:
        roots = np.array([], dtype=np.complex128)
    
    ax.axhline(0.0, linestyle="-.", linewidth=1, color="k")
    ax.axvline(0.0, linestyle="-.", linewidth=1, color="k")

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
    """ Plots poles ('x', red) and zeros ('o', blue) to one plot

    Args:
        poles (array): array of complex numbers, if None treated as empty array
        zeros (array): array of complex numbers, if None treated as empty array
        ax (Axes): matplotlib Axes object to plot to, default None, if None
            creates a new one
        **kwargs:
            ---

    Returns:
        ax (Axes): matplotlib Axes object
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
    """ Root plot with real and imaginary 0-level curves

    Args:
        roots (array): array of complex roots
        ax (Axes): matplotlib Axes object, if None new figure, ax is created
        meta (QpmrInfo): metadata obtained via qpmr(.)
    
    Returns:
        ax (Axes): matplotlib Axes object
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
    """ TODO """

    alpha=0.3
    color="red"

    if ax is None:
        _, ax = plt.subplots()

    for r in roots:
        circle = patches.Circle((r.real, r.imag), radius=ds/20., alpha=alpha, color=color)
        ax.add_patch(circle)
    
    return ax



