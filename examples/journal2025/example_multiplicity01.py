"""

"""

import numpy as np
from qpmr.quasipoly import examples
from qpmr.quasipoly import eval


coefs, delays = examples.self_inverse_polynomial(-1, radius=0.01, degree=6)
region = (-1.1, 0.9, -0.1, 0.1)
ds = 0.08

# solve grid
bmin, bmax = region[0] - 3*ds, region[1] + 3*ds
wmin, wmax = region[2] - 3*ds, region[3] + 3*ds

# construct grid, add to metadata - grid is cached
real_range = np.arange(bmin, bmax, ds)
imag_range = np.arange(wmin, wmax, ds)
complex_grid = 1j*imag_range.reshape(-1, 1) + real_range
re_grid = complex_grid.real
im_grid = complex_grid.imag

z_value = eval(coefs, delays, complex_grid)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.contour(re_grid, im_grid, np.real(z_value), levels=[0],
            colors='blue', alpha=0.5)
ax.contour(re_grid, im_grid, np.imag(z_value), levels=[0],
            colors='green', alpha=0.5)

ax.set_xlabel(r"$\Re (\lambda)$")
ax.set_ylabel(r"$\Im (\lambda)$")

plt.show()