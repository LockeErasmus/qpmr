"""
Delay distribution plots
------------------------
"""

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from . import utils

from qpmr.distribution.envelope_curve import _spectral_norms, _envelope_eval, _envelope_real_axis_crossing

def spectrum_distribution_diagram(x, y, mask, ax: Axes=None) -> Axes:
    """ Plots delay distribution

    Args:
        x (array): vector of theta_i = tau_max - tau_i sorted in ascending order
        y (array): vector of correspoding polynomial orders (associated to theta_i from `x`)
        mask (array): mask defining envelope, same length as `x` and `y`
        ax (Axes): matplotlib Axes object, if None new is created

    Returns:
        ax (Axes): matplotlib Axes object
    """
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(x[~mask],y[~mask], "o", color="blue")
    ax.plot(x[mask], y[mask], "o-", color="blue")
    
    ax.yaxis.grid(True)
    ax.set_xlabel(r"$\vartheta_i = \tau_{max} - \tau_i $")
    ax.set_ylabel(r"degree of $p_i$")

    return ax

def chain_asymptotes(mi: npt.NDArray, abs_wk: list[npt.NDArray], region: tuple, ax: Axes=None) -> Axes:
    """ TODO
    
    TODO allow custom beta
    beta = np.linspace(region[0], region[1], 1000)
    """
    if ax is None:
        _, ax = plt.subplots()

    beta = np.linspace(region[0], region[1], 1000)
    for m, w in zip(mi, abs_wk):
        for ww in w: 
            plt.plot(beta, ww*np.exp(-1/m*beta), "k-", alpha=0.5)
    
    ax.set_xlabel(r"$\Re (\lambda)$")
    ax.set_ylabel(r"$\Im (\lambda)$")
    ax.set_xlim(region[0], region[1])
    ax.set_ylim(region[2], region[3])
    return ax

@utils.matplotlib_axes_default
def spectrum_envelope(norms: npt.NDArray, delays: npt.NDArray, region: tuple, ax: Axes) -> Axes:
    
    re_max = _envelope_real_axis_crossing(norms, delays)
    
    x = np.linspace(region[0], min(region[1], re_max), 1000)
    y = _envelope_eval(x, norms, delays)

    ax.plot(x, y, color="blue", alpha=0.5)

    ax.set_xlabel(r"$\Re (\lambda)$")
    ax.set_ylabel(r"$\Im (\lambda)$")
    ax.set_xlim(region[0], region[1])
    ax.set_ylim(region[2], region[3])
    return ax

    