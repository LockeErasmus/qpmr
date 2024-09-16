"""
Example 3
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
    from qpmr import qpmr

    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.getLogger("PIL").setLevel(logging.ERROR)
    logger = logging.getLogger("qpmr")
    logging.basicConfig(level=logging.DEBUG)

    roots, meta = qpmr(region, coefs, delays)

    logger.info(f"matlab number of roots: {len(matlab_roots)}, python: {len(roots)}")

    complex_grid = meta.complex_grid
    value = meta.z_value
    plt.figure()

    plt.subplot(121)
    plt.contour(np.real(complex_grid), np.imag(complex_grid), np.real(value), levels=[0], colors='blue')
    plt.contour(np.real(complex_grid), np.imag(complex_grid), np.imag(value), levels=[0], colors='green')
    plt.scatter(np.real(roots), np.imag(roots), marker="o", color="r")

    plt.subplot(122)
    plt.scatter(np.real(roots), np.imag(roots), marker="o", color="r", alpha=0.4)
    plt.scatter(np.real(matlab_roots), np.imag(matlab_roots), marker="x", color="b")

    #plt.figure()
    #plt.contour(np.real(complex_grid), np.imag(complex_grid), np.real(h(complex_grid)), levels=[0], colors='blue')
    #plt.contour(np.real(complex_grid), np.imag(complex_grid), np.imag(h(complex_grid)), levels=[0], colors='green')

    plt.show()
