"""
Example 4

qp = (1/4)s**(2.5) + s + (1/4)exp(-1.5s)(2.2*s + 2.8896)

qp = (1/4)s**(2.5) + s + (1/4)exp(-1.5s)(2*s + 2.8896)


"""
import logging
import numpy as np

region = [-1, 20, 0, 20]
delays = np.array([0, 1.5])
coefs = np.array([[0, 1., 0.25],
                  [2.8896/4, 2.2/4, 0]])
powers = np.array([0.0, 1.0, 2.5])

delays = np.array([0, 1.5])
coefs = np.array([[0, 1., 0.25],
                  [2.8896/4, 2.09/4, 0]])
powers = np.array([0.0, 1.0, 2.5])

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from qpmr.qpmr_v2_fractional import qpmr_fractional

    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.getLogger("PIL").setLevel(logging.ERROR)
    logger = logging.getLogger("qpmr")
    logging.basicConfig(level=logging.DEBUG)

    roots, meta = qpmr_fractional(region, coefs, delays, powers, grid_nbytes_max=None)

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
