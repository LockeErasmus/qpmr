"""
Example 6
"""

import numpy as np

region = [-10, 5, 0, 300]
coefs = np.array([[1.5, 0.2, 0, 20.1],
                  [1, 0, -2.1, 0],
                  [0, 0, 3.2, 0],
                  [0, 0, 0, 1.4]])

# coefs = np.array([[10.5, 0.2, 0, 20.1],
#                   [1, 0, -2.1, 0],
#                   [-0.5, 0, 3.2, 0],
#                   [0, 0, 0, 1.4]])

coefs = coefs[:,::-1] # powers of s are same as index here
delays=np.array([0, 1.3, 3.5, 4.3])

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import qpmr
    import qpmr.plot

    logger = qpmr.init_logger(level="DEBUG", format="%(name)s - %(message)s")

    roots, meta = qpmr.qpmr(region, coefs, delays)
    thetas, degrees, mask = qpmr.distribution_diagram(coefs, delays)
    mi, abs_wk, _, = qpmr.chain_asymptotes(coefs, delays)
    sol = qpmr.spectral_abscissa_diff(coefs, delays)
    
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(8,5))
    qpmr.plot.qpmr_basic(roots, meta, ax=ax1)
    qpmr.plot.roots_basic(roots, ax=ax2)
    
    fig, ax = plt.subplots(1,1,figsize=(8,5))
    qpmr.plot.chain_asymptotes(mi, abs_wk, region, ax=ax)
    ax.axvline(sol.root, color="blue")
    qpmr.plot.roots_basic(roots, ax=ax)
    
    fig, ax = plt.subplots(1,1,figsize=(8,3))
    qpmr.plot.delay_distribution_basic(thetas, degrees, mask, ax=ax)
    plt.show()
