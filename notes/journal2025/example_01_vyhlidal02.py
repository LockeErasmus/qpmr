"""
Introductory example for the journal article
--------------------------------------------

"""
# %%
import qpmr
import qpmr.plot
import qpmr.quasipoly.examples as examples
import matplotlib.pyplot as plt
import numpy as np

qpmr.init_logger("DEBUG")

coefs, delays = examples.vyhlidal2014qpmr(2)
region = (-6, 3, 0, 200)
roots, info = qpmr.qpmr(coefs, delays, region=region)

th, deg, m = qpmr.distribution_diagram(coefs, delays)
mu, abs_wk = qpmr.chain_asymptotes(coefs, delays)

# %%
import settings
fig, ax = plt.subplots(1, 1, figsize=(settings.LINE_WIDTH, settings.MEDIUM_HEIGHT))
qpmr.plot.chain_asymptotes(mu, abs_wk, region=info.region, ax=ax)
qpmr.plot.roots(roots, ax=ax)
settings.save_figure(fig, "example_02_vyhlidal_roots")

# %%
fig, ax = plt.subplots(1, 1, figsize=(settings.LINE_WIDTH,settings.SMALL_HEIGHT))
qpmr.plot.spectrum_distribution_diagram(th, deg, m, ax=ax)
settings.save_figure(fig, "example_02_vyhlidal_spectrum_distribution_diagram")