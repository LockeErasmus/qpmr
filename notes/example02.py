"""
Example 2
"""
import numpy as np

region = [-4.5, 2.5, 0, 100]
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
    import qpmr
    import qpmr.plot

    logger = qpmr.init_logger(level="DEBUG")

    roots, meta = qpmr.qpmr(region, coefs, delays)
    complex_grid = meta.complex_grid
    value = meta.z_value
    
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(8,5))
    qpmr.plot.qpmr_contour(roots, meta, ax=ax1)
    qpmr.plot.roots(roots, ax=ax2)
    plt.show()
