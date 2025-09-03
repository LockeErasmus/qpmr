"""

"""
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import numpy.typing as npt

from qpmr.qpmr_metadata import QpmrRecursionContext
from . import basic



def qpmr_solution_tree(ctx: QpmrRecursionContext, ax: Axes):
    """ """

    if ax is None:
        _, ax = plt.subplots()

    for leaf in ctx.solution_tree.leaves:
        a,b,c,d = leaf.region
        
        if False: # TODO, failed leaf
            ... # TODO

        ax.add_patch(
            patches.Rectangle((a, c), b-a, d-c, linewidth=1, edgecolor='black', facecolor='none')
        )
        basic.qpmr_contour(leaf.roots, leaf, ax=ax)        
        basic.argument_principle_circle(leaf.roots, leaf.ds, ax=ax)

    basic.roots(ctx.roots, ax=ax)
    return ax



