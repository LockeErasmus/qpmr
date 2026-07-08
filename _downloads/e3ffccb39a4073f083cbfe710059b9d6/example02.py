r"""
Example 2
=========

We consider neutral quasi-polynomial

.. math::

    h(s) = s^4 + 0.2s^3 + 5s + 2.1 + (0.5s^4 - 2.1s) e^{-1.5s} + (0.3s^4 + 3.2s^2) e^{-2.2s} + (1.2s^2) e^{-4.3s} + 3 e^{-6.3s},

and in this example, we will:

1. compute the spectrum of decisive zeros,
2. compute distribution diagram.
3. compute vertical strip of neutral zeros,
4. compute asymptotic exponential chains of zeros,
5. plot all of the above
"""

# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import qpmr
import qpmr.plot


delays = np.array([0.0, 1.5, 2.2, 4.3, 6.3])
coefs = np.array([[2.1, 5, 0, 0.2, 1],
                  [0, -2.1, 0, 0, 0.5],
                  [0, 0, 3.2, 0, 0.3],
                  [0, 0, 1.2, 0, 0,],
                  [3, 0, 0, 0, 0]])


roots, info = qpmr.qpmr(coefs, delays)
th, deg, mask = qpmr.distribution_diagram(coefs, delays)
cdm, cdp = qpmr.bounds_neutral_strip(coefs, delays)
mi, abs_wk = qpmr.chain_asymptotes(coefs, delays)

# %%
# Plot decisive zeros, neutral strip and asymptotic exponentials

fig, ax = plt.subplots()
ax.axvline(cdm, color="blue")
ax.axvline(cdp, color="blue")
qpmr.plot.chain_asymptotes(mi, abs_wk, region=info.region, ax=ax)
qpmr.plot.roots(roots, ax=ax, label="roots")