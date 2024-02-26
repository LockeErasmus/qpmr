"""
"""
import logging

import matplotlib.pyplot as plt
import numpy as np

from qpmr import qpmr


_ = logging.getLogger("matplotlib").setLevel(logging.ERROR)
_ = logging.getLogger("PIL").setLevel(logging.ERROR)


if __name__ == "__main__":
    logger = logging.getLogger("qpmr")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


    
    region = [-10, 2, 0, 30]
    # region = [1.4, 2.6, 1.0, 4.0] # region with no real 0-level contours
    # region = [-8, -4, 0, 30] # contours present, no crossings -> no roots

    delays = np.array([0.0, 1.0])
    coefs = np.array([[0, 1],[1, 0]])

    roots, meta = qpmr(region, coefs, delays)
    matlab_roots = np.array([-0.3181 + 1.3372j,
                             -2.0623 + 7.5886j,
                             -2.6532 +13.9492j,
                             -3.0202 +20.2725j,
                             -3.2878 +26.5805j,])

    complex_grid = meta.complex_grid
    value = meta.z_value
    
    plt.figure()

    plt.subplot(121)
    plt.contour(np.real(complex_grid), np.imag(complex_grid), np.real(value), levels=[0], colors='blue')
    plt.contour(np.real(complex_grid), np.imag(complex_grid), np.imag(value), levels=[0], colors='green')
    plt.scatter(np.real(roots), np.imag(roots), marker="o", color="r")

    plt.subplot(122)
    plt.scatter(np.real(roots), np.imag(roots), marker="o", color="r")
    plt.scatter(np.real(matlab_roots), np.imag(matlab_roots), marker="x", color="b")
    plt.show()
