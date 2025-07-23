"""
Example 7
---------
Wilkinson-like quasipolynomial
"""

import numpy as np

region = [-20, 20, 0, 30]

d = 5
expected_roots = np.arange(1, d, 1, dtype=np.complex128)
poly_coefs = np.polynomial.polynomial.polyfromroots(expected_roots)[np.newaxis, :]
coefs = np.r_[poly_coefs, poly_coefs]
delays = np.array([0., 1])

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import qpmr
    import qpmr.plot

    logger = qpmr.init_logger(level="DEBUG", format="%(name)s - %(message)s")

    roots, meta = qpmr.qpmr(region, coefs, delays, ds=0.5, numerical_method_kwargs={"tolerance":0.1})
    thetas, degrees, mask = qpmr.distribution_diagram(coefs, delays)
    mi, abs_wk = qpmr.chain_asymptotes(coefs, delays)

    fig, ax = plt.subplots(1,1,figsize=(8,5))
    #qpmr.plot.roots(roots, ax=ax)
    ax.scatter(meta.roots0.real, meta.roots0.imag, marker="o", s=10, facecolors='k', label="initial")
    ax.scatter(meta.roots_numerical.real, meta.roots_numerical.imag, marker="o", s=50, edgecolors="r", facecolors='none', label="after correction")
    for x0, xstar in zip(meta.roots0, meta.roots_numerical):
        ax.plot([x0.real, xstar.real],[x0.imag, xstar.imag], color="gray", alpha=0.3, zorder=0)
        #ax.arrow(x0.real, x0.imag, xstar.real - x0.real, xstar.imag - x0.imag, color="gray", zorder=0)

    print(meta.roots0)
    print(meta.roots_numerical)

    #ax.set_ylim(-2e-16, 2e-16)
    ax.scatter(expected_roots.real, expected_roots.imag, marker="o", s=100, edgecolors="b", facecolors='none', label="prescribed real zeros")

    ax.legend()

    fig, ax = plt.subplots(1,1,figsize=(8,5))
    qpmr.plot.qpmr_contour(roots, meta, ax=ax)
    ax.scatter(meta.roots0.real, meta.roots0.imag, marker="o", s=10, facecolors='k', label="initial", zorder=0)
    ax.legend()


    plt.show()
