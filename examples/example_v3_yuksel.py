"""

"""

import qpmr
from qpmr.qpmr_v3 import qpmr as qpmr_v3
import qpmr.quasipoly.examples as examples


coefs, delays = examples.yuksel2023distributed()
region = (-12, 1, -0.1, 5000)
region = (-12, 1, -0.1, 500)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import qpmr
    import qpmr.plot

    from qpmr.qpmr_v3 import qpmr as qpmr_v3

    logger = qpmr.init_logger(level="DEBUG")

    roots, ctx = qpmr_v3(region, coefs, delays, multiplicity_heuristic=True)    
    
    print(ctx.render_tree)

    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(8,5))
    # qpmr.plot.qpmr_contour(roots, meta, ax=ax1)
    
    for leaf in ctx.solution_tree.leaves:
        a,b,c,d = leaf.region
        rectangle = patches.Rectangle((a, c), b-a, d-c, linewidth=1, edgecolor='black', facecolor='none')
        # Add the rectangle to the axes

        qpmr.plot.qpmr_contour(leaf.roots, leaf, ax=ax1)
        ax1.add_patch(rectangle)
        qpmr.plot.argument_principle_circle(leaf.roots, leaf.ds, ax=ax1)


    
    qpmr.plot.roots(roots, ax=ax2)
    qpmr.plot.roots(roots, ax=ax1)
    
    
    
    plt.show()
