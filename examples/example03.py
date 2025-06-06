"""
Example 3
"""
import numpy as np

region = [-6, 2, 0, 200]
delays = np.array([0.0, 1.5, 2.2, 4.3, 6.3])
coefs = np.array([[2.1, 5, 0, 0.2, 1],
                  [0, -2.1, 0, 0, 0.5],
                  [0, 0, 3.2, 0, 0.3],
                  [0, 0, 1.2, 0, 0,],
                  [3, 0, 0, 0, 0]])

matlab_roots = np.array([], dtype=np.complex128)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import qpmr
    import qpmr.plot

    logger = qpmr.init_logger(level="DEBUG", format="%(name)s - %(message)s")

    roots, meta = qpmr.qpmr(region, coefs, delays)
    thetas, degrees, mask = qpmr.distribution_diagram(coefs, delays)
    mi, abs_wk = qpmr.chain_asymptotes(coefs, delays)
    
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(8,5))
    qpmr.plot.qpmr_basic(roots, meta, ax=ax1)
    qpmr.plot.roots_basic(roots, ax=ax2)
    ax2.scatter(matlab_roots.real, matlab_roots.imag, marker="o", s=80, edgecolors="b", facecolors='none', label="matlab")
    
    fig, ax = plt.subplots(1,1,figsize=(8,5))
    qpmr.plot.chain_asymptotes(mi, abs_wk, region, ax=ax)
    qpmr.plot.roots_basic(roots, ax=ax)
    

    fig, ax = plt.subplots(1,1,figsize=(8,3))
    qpmr.plot.delay_distribution_basic(thetas, degrees, mask, ax=ax)

    plt.show()
