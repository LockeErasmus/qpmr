"""
Basic root plots
----------------
"""

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from qpmr import QpmrOutputMetadata

def roots_basic(roots: npt.NDArray, ax: matplotlib.axes.Axes=None, **kwargs):
    """ Plots roots into complex plane
    
    Args:
        roots (array): array of complex roots
        ax (Axes): matplotlib Axes object, if None new figure, ax is created
        **kwargs:
            tol (float): tolerance for assuming Re(root) ~ 0, defgault 1e-10

    Returns:
        ax (Axes): matplotlib Axes object
    """

    tol = kwargs.get("tol", 1e-10)

    if ax is None:
        _, ax = plt.subplots()
    
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
    
def qpmr_basic(roots: npt.NDArray, meta:QpmrOutputMetadata=None, ax: matplotlib.axes.Axes=None, **kwargs):
    """ Root plot with real and imaginary 0-level curves

    Args:
        roots (array): array of complex roots
        ax (Axes): matplotlib Axes object, if None new figure, ax is created
        meta (QpmrOutputMetadata): metadata obtained via qpmr(.)
    
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
                colors='blue', alpha=0.5, linewidth=0.5)
    ax.contour(re_grid, im_grid, np.imag(meta.z_value), levels=[0],
                colors='green', alpha=0.5, linewidth=0.5)
    
    roots_real = np.real(roots)
    roots_imag = np.imag(roots)

    ax.scatter(roots_real, roots_imag, marker="x", color="r", linewidths=0.5)
    
    ax.set_xlabel(r"$\Re (\lambda)$")
    ax.set_ylabel(r"$\Im (\lambda)$")
    
    return ax