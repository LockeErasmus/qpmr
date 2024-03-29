"""
"""
import logging

import matplotlib.pyplot as plt
import numpy as np

from qpmr import distribution_diagram
from qpmr.quasipoly import QuasiPolynomial

_ = logging.getLogger("matplotlib").setLevel(logging.ERROR)
_ = logging.getLogger("PIL").setLevel(logging.ERROR)
logger = logging.getLogger("qpmr")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    region = [-4.5, 4, 0, 100]
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

    qp = QuasiPolynomial(coefs, delays)
    x, y, mask = distribution_diagram(qp)

    # x = np.arange(10)
    # y = np.arange(10)
    # mask = np.full_like(x, fill_value=False, dtype=bool)
    # concave_envelope_inplace(x,y,mask)

    if True:

        plt.figure()
        plt.plot(x,y, "ro")
        plt.plot(x[mask], y[mask], "x-")
        
        
        #plt.plot(qp.delays, qp.degree - qp.poly_degrees, "o")      

        #plt.figure()
        #plt.contour(np.real(complex_grid), np.imag(complex_grid), np.real(h(complex_grid)), levels=[0], colors='blue')
        #plt.contour(np.real(complex_grid), np.imag(complex_grid), np.imag(h(complex_grid)), levels=[0], colors='green')

        plt.show()
