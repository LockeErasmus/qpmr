"""

"""
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import numpy.typing as npt

from . import basic



def qpmr_solution_tree(ctx , ax: Axes, **kwargs):
    """Visualize the QPmR solution tree and regional subdivisions.

    Parameters
    ----------
    ctx
        Context object with ``solution_tree`` and ``roots`` attributes.
    ax : Axes
        Matplotlib axes to draw on.

    Returns
    -------
    ax : Axes
        Matplotlib axes with the tree visualization.
    """
    if ax is None:
        _, ax = plt.subplots()
    for leaf in ctx.solution_tree.leaves:
        a,b,c,d = leaf.region
        if leaf.status == "SOLVED":
            if isinstance(leaf.status_message, str) and leaf.status_message.startswith("ARGP"):
                ax.add_patch(
                    patches.Rectangle((a, c), b-a, d-c, linewidth=0, fill=None, hatch='///')
                )
            else:
                ax.add_patch(
                    patches.Rectangle((a, c), b-a, d-c, linewidth=1, edgecolor='black', facecolor='none')
                )
                basic.qpmr_contour(leaf.roots, leaf, ax=ax)
        
            if True: # argument principle heuristic was turn on TODO
                basic.argument_principle_circle(leaf.roots, leaf.ds, ax=ax)
        
        elif leaf.status == "FAILED":
            # ax.add_patch(
            #     patches.Rectangle((a, c), b-a, d-c, linewidth=0, fill="", hatch='///')
            # )
            pass
        else: # UNPROCESSED
            pass


    basic.roots(ctx.roots, ax=ax)
    return ax



