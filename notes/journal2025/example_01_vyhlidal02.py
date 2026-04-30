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
region = (-4, 3, 0, 100)
region = (-6, 3, 0, 100)
roots, info = qpmr.qpmr(coefs, delays, region=region)

th, deg, mask = qpmr.distribution_diagram(coefs, delays)
mu, abs_wk = qpmr.chain_asymptotes(coefs, delays)

# %%
import settings
fig, ax = plt.subplots(1, 1, figsize=(settings.LINE_WIDTH, settings.LARGE_HEIGHT))

ax.grid(True, linewidth=0.5)

# ax.axhline(0.0, linestyle="-.", linewidth=0.5, color="k")
# ax.axvline(0.0, linestyle="-.", linewidth=0.5, color="k")

beta = np.linspace(region[0], region[1], 1000)
for mui, w in zip(mu, abs_wk):
    for ww in w: 
        plt.plot(beta, ww*np.exp(-1/mui*beta), "r-", alpha=1.0, linewidth=0.5)

ax.scatter(np.real(roots),
            np.imag(roots),
            marker="o",
            color="k",
            s=6,
            linewidths=0.5,
)

ax.set_xlabel(r"$\Re (\lambda)$")
ax.set_ylabel(r"$\Im (\lambda)$")
ax.set_xlim(region[0], region[1])
ax.set_ylim(region[2], region[3])

qpmr.plot.roots(roots, ax=ax)
settings.save_figure(fig, "example_02_vyhlidal_roots")

# %%
fig, ax = plt.subplots(1, 1, figsize=(settings.LINE_WIDTH,settings.SMALL_HEIGHT))
ax.grid(True, linewidth=0.5)

ax.plot(th[~mask],deg[~mask], "o", color="blue", markersize=4)
ax.plot(th[mask], deg[mask], "o-", color="blue", markersize=4, linewidth=0.5)

ax.set_xlabel(r"$\vartheta_i$")
ax.set_ylabel(r"$m_i$")

settings.save_figure(fig, "example_02_vyhlidal_spectrum_distribution_diagram")