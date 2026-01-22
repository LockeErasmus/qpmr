"""
"""
import logging

import matplotlib.pyplot as plt
import numpy as np

import qpmr
#from qpmr import distribution_diagram
# from qpmr.quasipoly import QuasiPolynomial

if __name__ == "__main__":
    logger = qpmr.init_logger(level="DEBUG", format="%(name)s - %(message)s")
    
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
    
    x, y, mask = qpmr.distribution_diagram(coefs, delays)

    import qpmr.plot

    qpmr.plot.spectrum_distribution_diagram(x, y, mask)

    plt.show()


