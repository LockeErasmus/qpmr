"""
Experimental complex plane plots
--------------------------------
"""

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

def experimental(roots: npt.NDArray, ax: Axes=None, **kwargs):
    """Experimental polar-style root plot.

    Parameters
    ----------
    roots : ndarray
        Complex roots.
    ax : Axes, optional
        Matplotlib axes. Created if ``None``.
    tol : float, optional
        Tolerance for stability classification (default 1e-10).

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

    ax.axvline(np.pi / 2, linestyle="-.", linewidth=1, color="k") # stability boundary

    x = np.abs(np.angle(roots))
    y = np.sign(roots.imag) * np.abs(roots)

    mask_negative = roots.real < -tol
    if np.any(mask_negative) > 0:
        ax.scatter(x[mask_negative],
                   y[mask_negative],
                   marker="x",
                   color="g",
                   linewidths=0.5,
        )
    
    mask_positive = roots.real > tol
    if np.any(mask_positive) > 0:
        ax.scatter(x[mask_positive],
                   y[mask_positive],
                   marker="x",
                   color="r",
                   linewidths=0.5,
        )

    mask_zero = ~(mask_positive | mask_negative)
    if np.any(mask_zero) > 0:
        ax.scatter(x[mask_zero],
                   y[mask_zero],
                   marker="x",
                   color="b",
                   linewidths=0.5,
        )

    ax.set_xlabel(r"$\|arg(\lambda)\|$")
    ax.set_ylabel(r"$sign(\Im \lambda )\|\lambda\|")

    return ax