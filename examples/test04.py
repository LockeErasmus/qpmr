"""
"""
import logging

import matplotlib.pyplot as plt
import numpy as np

from qpmr import qpmr, distribution_diagram
from qpmr.quasipoly import QuasiPolynomial


_ = logging.getLogger("matplotlib").setLevel(logging.ERROR)
_ = logging.getLogger("PIL").setLevel(logging.ERROR)
logger = logging.getLogger("qpmr")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    L = 0.38123
    Tz = 63.031
    Tp = 1/0.63304
    kp = -2.
    kd = -2.

    region = [100, 800, -0.1, 0.1]

    for tau in np.arange(0.5, 0.003, -0.003):
        delays = np.array([0, tau])
        coefs = np.array([
            [L*Tp*(kp+kd/tau), Tp+L*Tz*Tp*(kp+kd/tau),1.0],
            [-L*Tp*kd/tau, -L*Tz*Tp*kd/tau, 0],
        ])
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

    #plt.figure()
    #plt.contour(np.real(complex_grid), np.imag(complex_grid), np.real(h(complex_grid)), levels=[0], colors='blue')
    #plt.contour(np.real(complex_grid), np.imag(complex_grid), np.imag(h(complex_grid)), levels=[0], colors='green')

    plt.show()
        

