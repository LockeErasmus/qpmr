"""
"""
import logging

import matplotlib.pyplot as plt
import numpy as np

from qpmr.qpmr_v2 import qpmr


_ = logging.getLogger("matplotlib").setLevel(logging.ERROR)
_ = logging.getLogger("PIL").setLevel(logging.ERROR)
logger = logging.getLogger("qpmr")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    region = [-10, 2, 0, 30]
    delays = np.array([0.0, 1.0])
    coefs = np.array([[0, 1],[1, 0]])

    roots, meta = qpmr(region, coefs, delays)

    def h(s):
        return s + np.exp(-s)

    if True:
        complex_grid = meta.get("complex_grid")
        value = meta.get("func_value")
        plt.figure()
        plt.contour(np.real(complex_grid), np.imag(complex_grid), np.real(value), levels=[0], colors='blue')
        plt.contour(np.real(complex_grid), np.imag(complex_grid), np.imag(value), levels=[0], colors='green')
        plt.scatter(np.real(roots), np.imag(roots), marker="o", color="r")
        

        #plt.figure()
        #plt.contour(np.real(complex_grid), np.imag(complex_grid), np.real(h(complex_grid)), levels=[0], colors='blue')
        #plt.contour(np.real(complex_grid), np.imag(complex_grid), np.imag(h(complex_grid)), levels=[0], colors='green')

        plt.show()
