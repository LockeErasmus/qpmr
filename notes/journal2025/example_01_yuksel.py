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

coefs, delays = examples.yuksel2023distributed(True)
coefs_z, delays_z = examples.yuksel2023distributed(False)
region = (-10, 1, -10, 3000)

roots, info = qpmr.qpmr(coefs, delays, region=region)
zeros, info = qpmr.qpmr(coefs, delays, region=region)
th, deg, m = qpmr.distribution_diagram(coefs, delays)
mu, abs_wk = qpmr.chain_asymptotes(coefs, delays)

# %%
import settings
fig, ax = plt.subplots(1, 1, figsize=(settings.LINE_WIDTH, settings.LARGE_HEIGHT))
qpmr.plot.chain_asymptotes(mu, abs_wk, region=info.region, ax=ax)
# qpmr.plot.roots(roots, ax=ax)
qpmr.plot.pole_zero(roots, zeros, ax=ax)
settings.save_figure(fig, "example_01_yuksel_pole_zero")