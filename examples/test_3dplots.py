"""
Test for plotting real and imaginary plots as 3D
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
    
    region = [-10, 5, 0, 50]
    delays = np.array([0.0, 1.3, 3.5, 4.3])
    coefs = np.array([[20.1, 0, 0.2, 1.5],
                      [0, -2.1, 0, 1],
                      [0, 3.2, 0, 0],
                      [1.4, 0, 0, 0]])

    roots, meta = qpmr(region, coefs, delays)

    if True:
        complex_grid = meta.complex_grid
        value = meta.z_value
        real_value = np.tanh(np.real(value))
        
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        ax.plot_surface(np.real(complex_grid), np.imag(complex_grid), real_value,
                        rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        ax.set_title('surface')

        #plt.figure()
        #plt.contour(np.real(complex_grid), np.imag(complex_grid), np.real(h(complex_grid)), levels=[0], colors='blue')
        #plt.contour(np.real(complex_grid), np.imag(complex_grid), np.imag(h(complex_grid)), levels=[0], colors='green')

        plt.show()
