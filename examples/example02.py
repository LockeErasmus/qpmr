"""
Example 2
"""
import logging

import numpy as np

region = [-4.5, 2.5, 0, 300]
delays = np.array([24.99, 23.35, 19.9, 18.52, 13.32, 10.33, 8.52, 4.61, 0.0])
coefs = np.array([[51.7, 0, 0, 0, 0, 0, 0, 0 , 0],
                    [1.5, -0.1, 0.04, 0.03, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0.5, 0, 0, 0, 0, 0],
                    [0, 25.2, 0, -0.9, 0.2, 0.15, 0, 0, 0],
                    [7.2, -1.4, 0, 0, 0.1, 0, 0.8, 0, 0],
                    [0, 19.3, 2.1, 0, -8.7, 0, 0, 0, 0],
                    [0, 6.7, 0, 0, 0, -1.1, 0, 1, 0],
                    [29.1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, -1.8, 0.001, 0, 0, -12.8, 0, 1.7, 0.2]])

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from qpmr import qpmr

    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.getLogger("PIL").setLevel(logging.ERROR)
    logger = logging.getLogger("qpmr")
    logging.basicConfig(level=logging.DEBUG)
    
    roots, meta = qpmr(region, coefs, delays)
    complex_grid = meta.complex_grid
    value = meta.z_value

    plt.figure()

    plt.subplot(121)
    plt.contour(np.real(complex_grid), np.imag(complex_grid), np.real(value), levels=[0], colors='blue')
    plt.contour(np.real(complex_grid), np.imag(complex_grid), np.imag(value), levels=[0], colors='green')
    plt.scatter(np.real(roots), np.imag(roots), marker="o", color="r")

    plt.subplot(122)
    plt.scatter(np.real(roots), np.imag(roots), marker="o", color="r", alpha=0.4)

    plt.show()
