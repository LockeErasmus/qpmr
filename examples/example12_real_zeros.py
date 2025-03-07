"""
Example 11
"""

import numpy as np

region = [-20, 20, 0, 30]
expected_roots = np.arange(1, 5, 1, dtype=np.complex128)
poly_coefs = np.polynomial.polynomial.polyfromroots(expected_roots)[np.newaxis, :]
coefs = np.r_[poly_coefs, poly_coefs, poly_coefs, poly_coefs]
delays = np.array([0., 0.001, 2, 3])

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import qpmr
    import qpmr.plot

    logger = qpmr.init_logger(level="DEBUG", format="%(name)s - %(message)s")

    roots, meta = qpmr.qpmr(region, coefs, delays)   
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(8,5))
    qpmr.plot.qpmr_basic(roots, meta, ax=ax1)
    ax1.scatter(meta.roots0.real, meta.roots0.imag, marker="o", s=50, facecolors='k', label="initial")
    ax1.scatter(meta.roots_numerical.real, meta.roots_numerical.imag, marker="x", s=50, facecolors='k', label="after correction")
    ax1.set_ylim(-2e-16, 2e-16)

    qpmr.plot.roots_basic(roots, ax=ax2)
    ax2.scatter(expected_roots.real, expected_roots.imag, marker="o", s=80, edgecolors="b", facecolors='none', label="prescribed real zeros")

    plt.show()
