r"""
Example 1
=========

"""
import matplotlib.pyplot as plt
import numpy as np
import qpmr
import qpmr.plot

delays = np.array([0.0, 1.0])
coefs = np.array([[0, 1],[1, 0.]])
region = [-10, 2, 0, 30]

# %%
roots, info = qpmr.qpmr(coefs, delays, region=region)
print(roots)

qpmr.plot.roots(roots)
plt.show()

# fig, ax = plt.subplots(1,1,figsize=(8,5))
#     qpmr.plot.chain_asymptotes(mi, abs_wk, region, ax=ax)
#     qpmr.plot.roots(roots, ax=ax)
    

#     fig, ax = plt.subplots(1,1,figsize=(8,3))
#     qpmr.plot.spectrum_distribution_diagram(thetas, degrees, mask, ax=ax)
