"""
Example 1
"""
import logging

import numpy as np

region = [-10, 2, 0, 30]
# region = [1.4, 2.6, 1.0, 4.0] # region with no real 0-level contours
# region = [-8, -4, 0, 30] # contours present, no crossings -> no roots
delays = np.array([0.0, 1.0])
coefs = np.array([[0, 1],[1, 0]])
matlab_roots = np.array([-0.3181 + 1.3372j,
                         -2.0623 + 7.5886j,
                         -2.6532 +13.9492j,
                         -3.0202 +20.2725j,
                         -3.2878 +26.5805j,])

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import qpmr
    import qpmr.plot

    logger = qpmr.init_logger(level="DEBUG")

    roots, meta = qpmr.qpmr(region, coefs, delays)
    complex_grid = meta.complex_grid
    value = meta.z_value
    
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(8,5))
    qpmr.plot.qpmr_basic(roots, meta, ax=ax1)
    qpmr.plot.roots_basic(roots, ax=ax2)
    ax2.scatter(matlab_roots.real, matlab_roots.imag, marker="o", s=80, edgecolors="b", facecolors='none', label="matlab")
    plt.show()
