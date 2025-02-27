"""
Delay distribution plots
------------------------
"""

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

def delay_distribution_basic(x, y, mask, ax=None):
    """ Plots delay distribution

    Args:
        x (array): vector of tau_max - tau sorted in ascending
        y (array): order of according polynomial (associated to delay from x)
        ax (Axes): matplotlib Axes object, if None new figure, ax is created

    Returns:
        ax (Axes): matplotlib Axes object
    """
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(x[~mask],y[~mask], "ro", color="blue")
    ax.plot(x[mask], y[mask], "o-", color="blue")
    
    ax.yaxis.grid(True)
    ax.set_xlabel(r"$\vartheta_i = \tau_{max} - \tau_i $")
    ax.set_ylabel(r"degree of $p_i$")

    return ax

def chain_asymptotes(mi: npt.NDArray, abs_wk: list[npt.NDArray], region: tuple, ax=None):
    """ TODO
    
    TODO allow custom beta
    beta = np.linspace(region[0], region[1], 1000)
    """
    if ax is None:
        _, ax = plt.subplots()

    beta = np.linspace(region[0], region[1], 1000)
    for m, w in zip(mi, abs_wk):
        for ww in w: 
            plt.plot(beta, ww*np.exp(-1/m*beta),"k-", alpha=0.5)
    
    ax.set_xlabel(r"$\Re (\lambda)$")
    ax.set_ylabel(r"$\Im (\lambda)$")
    ax.set_xlim(region[0], region[1])
    ax.set_ylim(region[2], region[3])
    return ax
