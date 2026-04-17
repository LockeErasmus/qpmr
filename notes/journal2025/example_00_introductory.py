"""
Introductory example for the journal article
--------------------------------------------

"""

import qpmr
import qpmr.plot
import matplotlib.pyplot as plt
import numpy as np

delays = np.array([0, 1, 2.])
coefs = np.array([[0, 1],[1, 1],[1, 0.]])

roots, info = qpmr.qpmr(coefs, delays)
th, deg, m = qpmr.distribution_diagram(coefs, delays)
mu, abs_wk = qpmr.chain_asymptotes(coefs, delays)
cdp, cdm = qpmr.bounds_neutral_strip(coefs, delays)

# %%
import settings
fig, ax = plt.subplots(1, 1, figsize=(settings.LINE_WIDTH, settings.MEDIUM_HEIGHT))
ax.axvline(cdp, color="blue")
ax.axvline(cdm, color="blue")
qpmr.plot.chain_asymptotes(mu, abs_wk, region=info.region, ax=ax)
qpmr.plot.roots(roots, ax=ax)
settings.save_figure(fig, "example_00_roots")

# %%
fig, ax = plt.subplots(1, 1, figsize=(settings.LINE_WIDTH,settings.SMALL_HEIGHT))
qpmr.plot.spectrum_distribution_diagram(th, deg, m, ax=ax)
settings.save_figure(fig, "example_00_spectrum_distribution_diagram")