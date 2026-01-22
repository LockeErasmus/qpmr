"""
Example 11
"""

import numpy as np

import qpmr
from qpmr.quasipoly import multiply

import qpmr.zero_multiplicity



c = -0.1
degree = 6
scale = 10.
coefs = ( (scale * np.poly1d([1, -c])) ** degree - 1 ).coeffs[::-1]
coefs = coefs[None, :]
region = (-5,5,-5,5)
delays = np.array([0.])
ds=0.2

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import qpmr
    import qpmr.plot

    from qpmr.qpmr_v3 import qpmr as qpmr_v3

    logger = qpmr.init_logger(level="DEBUG", format="%(name)s - %(message)s")

    roots, meta = qpmr_v3(region, coefs, delays)

    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(12,6)) 
    ax1.add_patch(
        plt.Circle((c,0), 1/scale, color="r", alpha=0.2)
    )
    for r in roots:
        a, b = np.real(r), np.imag(r)
        ax1.add_patch(
            plt.Circle((a,b), ds/20, color="b", alpha=0.1)
        )

    qpmr.plot.roots(roots, ax=ax1)
    qpmr.plot.qpmr_contour(roots, meta, ax=ax2)

    plt.show()
