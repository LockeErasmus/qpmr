"""
Example 11
"""
import logging

import numpy as np

region = [0, 20, -1, 5]
expected_roots = np.arange(1, 20, 1, dtype=np.complex128)
coefs = np.polynomial.polynomial.polyfromroots(expected_roots)[np.newaxis, :]
delays = np.array([0.0])

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import qpmr
    import qpmr.plot

    logger = qpmr.init_logger(level="DEBUG", format="%(name)s - %(message)s")

    roots, meta = qpmr.qpmr(region, coefs, delays, numerical_method="newton", ds=0.012)   
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(8,5))
    qpmr.plot.qpmr_contour(roots, meta, ax=ax1)
    qpmr.plot.roots(roots, ax=ax2)
    ax2.scatter(expected_roots.real, expected_roots.imag, marker="o", s=80, edgecolors="b", facecolors='none', label="matlab")

    plt.show()
