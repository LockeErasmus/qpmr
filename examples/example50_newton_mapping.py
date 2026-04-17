"""
Newton Mapping Example
----------------------

"""
import matplotlib.pyplot as plt
import numpy as np

import qpmr
from matplotlib.colors import ListedColormap
import qpmr.plot
from qpmr.quasipoly import derivative, eval # TODO eval name
import qpmr.quasipoly.examples as examples


coefs, delays = examples.vyhlidal2014qpmr(1)
# coefs, delays = examples.vyhlidal2014qpmr(3)
dcoefs, ddelays = derivative(coefs, delays)

N = 1000
region = (-18, 2, 0, 20)
# region = (-2, 2, 0, 12)

roots, info = qpmr.qpmr(coefs, delays, region=region)

# Newton map

z = 1j*np.linspace(region[2], region[3], N).reshape(-1, 1) + np.linspace(region[0], region[1], N)
z0 = z.copy()

newton_steps = np.zeros_like(z, dtype=np.complex128)
iter_count = np.zeros_like(z, dtype=int)

max_iter = 20
tol = 1e-6

for i in range(max_iter):
    fz = eval(coefs, delays, z)
    dfz = eval(dcoefs, ddelays, z)

    mask = np.isnan(z) | (np.abs(dfz) < 1e-12)  # TODO tolerance for zero derivative
    z[mask] = np.nan
    newton_steps[~mask] = fz[~mask] / dfz[~mask]
    z[~mask] = z[~mask] - newton_steps[~mask]

    mask_converged = np.abs(newton_steps) < tol
    iter_count[~mask_converged] += 1
    print(f"Iteration {i+1}/{max_iter}, converged points: {np.sum(mask_converged)}")

# visualize

nroots = len(roots)

dist = np.abs(z[..., None] - roots)
closest_root = np.argmin(dist, axis=-1)


import matplotlib.pyplot as plt

cmap = plt.cm.get_cmap('Set1', nroots)
img = np.zeros((*z.shape, 3))

norm_iter = iter_count / max_iter

for k in range(nroots):
    mask = closest_root == k
    for c in range(3):
        img[mask, c] = cmap(k)[c] * (1. - norm_iter[mask])


fig, ax = plt.subplots()

ax.imshow(img, extent=[region[0], region[1], region[2], region[3]], 
            origin='lower', aspect='auto', interpolation='nearest',
            rasterized=True)
ax.scatter(roots.real, roots.imag, color='black', facecolors='white', 
            s=64, edgecolors='black')

ax.set_xlabel('β')
ax.set_ylabel('ω')
ax.set_title('Newton Basin Fractal')

plt.show()



