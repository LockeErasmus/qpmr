"""
Example 4
"""
import logging
import numpy as np

region = [-10, 5, 0, 50]
delays = np.array([0.0, 1.3, 3.5, 4.3])
coefs = np.array([[20.1, 0, 0.2, 1.5],
                    [0, -2.1, 0, 1],
                    [0, 3.2, 0, 0],
                    [1.4, 0, 0, 0]])
matlab_roots = np.array([-1.3134 + 0.0000j,
                        -0.4091 + 1.5397j,
                        -1.6802 + 0.0000j,
                        1.1363 + 2.2206j,
                        -0.6429 + 3.1243j,
                        -1.0866 + 5.4556j,
                        -0.2143 + 7.2156j,
                        -1.4891 + 8.4415j,
                        -3.9318 + 9.3232j,
                        -1.6677 +11.2869j,
                        -0.3018 +12.1177j,
                        -1.8893 +14.1579j,
                        -0.3217 +16.9128j,
                        -2.0672 +17.0048j,
                        -4.6409 +17.3486j,
                        -2.2049 +19.9195j,
                        -0.3009 +21.7462j,
                        -2.2803 +22.7487j,
                        -2.4767 +25.6000j,
                        -5.0925 +25.2751j,
                        -0.3070 +26.5922j,
                        -2.5032 +28.5239j,
                        -2.5777 +31.2876j,
                        -0.3152 +31.4191j,
                        -5.4285 +33.1742j,
                        -2.7411 +34.2292j,
                        -0.3093 +36.2482j,
                        -2.7058 +37.0844j,
                        -2.8489 +39.8584j,
                        -0.3088 +41.0865j,
                        -5.6913 +41.0624j,
                        -2.9079 +42.8371j,
                        -2.8930 +45.6153j,
                        -0.3132 +45.9189j,
                        -3.0587 +48.4707j,
                        -5.9046 +48.9392j,])


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import qpmr
    import qpmr.plot

    logger = qpmr.init_logger(level="DEBUG", format="%(name)s - %(message)s")

    roots, meta = qpmr.qpmr(region, coefs, delays)
    thetas, degrees, mask = qpmr.distribution_diagram(coefs, delays)
    mi, abs_wk = qpmr.chain_asymptotes(coefs, delays)
    
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(8,5))
    qpmr.plot.qpmr_contour(roots, meta, ax=ax1)
    qpmr.plot.roots(roots, ax=ax2)
    ax2.scatter(matlab_roots.real, matlab_roots.imag, marker="o", s=80, edgecolors="b", facecolors='none', label="matlab")
    
    fig, ax = plt.subplots(1,1,figsize=(8,5))
    qpmr.plot.chain_asymptotes(mi, abs_wk, region, ax=ax)
    qpmr.plot.roots(roots, ax=ax)
    

    fig, ax = plt.subplots(1,1,figsize=(8,3))
    qpmr.plot.spectrum_distribution_diagram(thetas, degrees, mask, ax=ax)

    plt.show()
