"""
Utility functions for plotting
"""

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Callable


def matplotlib_axes_default(f: Callable):
    """ TODO """
    def wrapper(*args, **kwargs):
        ax = kwargs.pop("ax", None)
        if ax is None:
            fig, ax = plt.subplot()
        else:
            if not isinstance(ax, Axes):
                raise ValueError(f"Keyword argumnet `ax` has to be instance of class `matplotlib.axes.Axes`")
        f(*args, ax=ax, **kwargs)
        return ax
    return wrapper
