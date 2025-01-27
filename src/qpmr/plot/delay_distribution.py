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

    ax.plot(x,y, "ro")
    ax.plot(x[mask], y[mask], "x-")
    
    ax.yaxis.grid(True)
    ax.set_xlabel(r"$\tau_{max} - \tau $")
    ax.set_ylabel(r"order")

    return ax
