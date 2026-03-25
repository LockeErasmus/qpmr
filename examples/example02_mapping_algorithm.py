"""
Mapping algorithm example
-------------------------

"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from qpmr.core.quasipolynomial import eval
from qpmr.quasipoly.examples import vyhlidal2014qpmr_02



coefs, delays = vyhlidal2014qpmr_02()
region = (-0.8, 0.8, 0, 3.0)

def calculate_contours(coefs, delays, region, ds):
    re_range = np.linspace(region[0], region[1], int((region[1] - region[0]) / ds))
    im_range = np.linspace(region[2], region[3], int((region[3] - region[2]) / ds))
    z = 1j * im_range.reshape(-1, 1) + re_range
    fz = eval(coefs, delays, z)
    return fz, re_range, im_range

# %% Visualize the progress as animation
ds_vector = 1. / np.arange(5, 100, 1)

fig, ax = plt.subplots()

def update(n):
    ax.clear()

    ax.set_xlabel(r"$\Re (\lambda)$")
    ax.set_ylabel(r"$\Im (\lambda)$")
    ax.set_xlim(region[0], region[1])
    ax.set_ylim(region[2], region[3])

    fz, re_range, im_range = calculate_contours(coefs, delays, region, ds_vector[n])

    cont_real = ax.contour(re_range, im_range, np.real(fz), levels=[0], colors='blue')
    cont_imag = ax.contour(re_range, im_range, np.imag(fz), levels=[0], colors='red')

    ax.set_title(f"Step {n + 1}, ds={ds_vector[n]}")

    return cont_real, cont_imag

ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(ds_vector),
    interval=500,
    blit=False
)

plt.show()
