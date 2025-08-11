"""
Example 11
"""

import numpy as np

import qpmr
from qpmr.quasipoly import multiply

import qpmr.zero_multiplicity



coefs = np.array([[1,0],[0,1.]])
delays = np.array([0, 1.])

coefs, delays = multiply(coefs, delays, coefs, delays)
coefs, delays = multiply(coefs, delays, coefs, delays)
# coefs, delays = multiply(coefs, delays, coefs, delays)
# coefs, delays = multiply(coefs, delays, coefs, delays)
print(coefs, delays)
region = [0.5, 5, 0.1, 20]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import qpmr
    import qpmr.plot

    from qpmr.qpmr_v3 import qpmr as qpmr_v3

    logger = qpmr.init_logger(level="DEBUG", format="%(name)s - %(message)s")

    roots, meta = qpmr_v3(region, coefs, delays, numerical_method_kwargs={"max_iterations": 100, "tolerance": 1e-2})

    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(8,5))
    qpmr.plot.qpmr_contour(roots, meta, ax=ax1)
    ax1.scatter(meta.roots0.real, meta.roots0.imag, marker="o", s=50, facecolors='k', label="initial")
    ax1.scatter(meta.roots_numerical.real, meta.roots_numerical.imag, marker="x", s=50, facecolors='k', label="after correction")
    ax1.legend()

    qpmr.plot.roots(roots, ax=ax2)

    plt.show()
