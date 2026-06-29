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

# roots, info = qpmr.qpmr(coefs, delays, region=region)
# zeros, info = qpmr.qpmr(coefs_z, delays_z, region=region)
th, deg, m = qpmr.distribution_diagram(coefs, delays)
th_z, deg_z, m_z = qpmr.distribution_diagram(coefs_z, delays_z)
mu, abs_wk = qpmr.chain_asymptotes(coefs, delays)

# %%
import settings
# fig, ax = plt.subplots(1, 1, figsize=(settings.LINE_WIDTH, settings.LARGE_HEIGHT))
# qpmr.plot.chain_asymptotes(mu, abs_wk, region=info.region, ax=ax)
# # qpmr.plot.roots(roots, ax=ax)
# qpmr.plot.pole_zero(roots, zeros, ax=ax)
# settings.save_figure(fig, "example_01_yuksel_pole_zero")

# %%
fig, ax = plt.subplots(1, 1, figsize=(settings.LINE_WIDTH, settings.SMALL_HEIGHT))
ax.grid(True, linewidth=0.5)

ax.plot(th[~m],deg[~m], "o", color="blue", markersize=2)
ax.plot(th[m], deg[m], "o-", color="blue", markersize=2, linewidth=0.5)

ax.set_xlabel(r"$\vartheta_i$")
ax.set_ylabel(r"$m_i$")

ylim = ax.get_ylim()
plt.show()

settings.save_figure(fig, "example_01_yuksel_pole_zero_spectrum_distribution_diagram_poles", bbox_inches=None)

# %%
fig, ax = plt.subplots(1, 1, figsize=(settings.LINE_WIDTH, settings.SMALL_HEIGHT))
ax.grid(True, linewidth=0.5)

ax.plot(th_z[~m_z],deg_z[~m_z], "o", color="blue", markersize=2)
ax.plot(th_z[m_z], deg_z[m_z], "o-", color="blue", markersize=2, linewidth=0.5)

ax.set_xlabel(r"$\vartheta_i$")
ax.set_ylabel(r"$m_i$")
ax.set_ylim(ylim)

plt.show()
settings.save_figure(fig, "example_01_yuksel_pole_zero_spectrum_distribution_diagram_zeros", bbox_inches=None)
