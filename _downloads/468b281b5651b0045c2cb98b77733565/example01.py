r"""
Example 1
=========

We start with an introductory example of a neutral quasi-polynomial

.. math::

    h(s) = s + (1+s)e^{-s} + e^{-2s},

obtained as the product of a retarded quasi-polynomial 

.. math::

    h_r(s)=s+e^{-s},

and a neutral quasi-polynomial

.. math::

    h_n(s) = 1+e^{-s}

In this example, we will:

1. show how to define a quasi-polynomial in a matrix-vector format
2. construct and plot spectrum distribution diagram,
3. compute and plot the spectrum of decisive zeros.
"""

# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import qpmr
import qpmr.plot

# define quasi-polynomial as a matrix of coefficients and vector of delays
coefs = np.array([[0., 1],[1, 1], [1, 0]])
delays = np.array([0., 1, 2])

# %%
# We cancalculate and visualise spectrum distribution diagram of the
# quasi-polynomial. We expect to see two segments. One segment with positive
# slope corresponding to the retarded chain of zeros and one segment with zero
# slope corresponding to the neutral chain of zeros.

th, deg, mask = qpmr.distribution_diagram(coefs, delays)
qpmr.plot.spectrum_distribution_diagram(th, deg, mask)

# %%
# Now move to the computation of the spectrum of decisive zeros via QPmR
# algorithm.

roots, meta = qpmr.qpmr(coefs, delays)
print(f"meta.region: {meta.region}")
qpmr.plot.roots(roots)

# %%
# In this specific example, we can calculate zeros analytically. The zeros of 
# the retarded quasi-polynomial are given by 
# 
# .. math::
# 
#   s_k = W_k(-1), k\in\mathbb{Z},
# 
# where :math:`W_k` denotes the :math:`k-`th branch of the Lambert :math:`W` 
# function.
# 
# The zeros of the neutral quasi-polynomial are given by
# 
# .. math::
# 
#   s_k = -j(2k+1)\pi, k\in\mathbb{Z}.
# 
# Please note that for the Lambert :math:`W` function, we can use the
# implementation from the `scipy.special` package.
import warnings
try:
    from scipy.special import lambertw
    retarded_zeros = lambertw(-1, k=np.arange(0, 50))
except ImportError:
    warnings.warn("scipy.special.lambertw not available, install scipy to" \
    "compute retarded zeros")
    retarded_zeros = np.array([], dtype=complex)

neutral_zeros = np.array([1j * np.pi * (2 * k + 1) for k in range(0, 50)])

fig, ax = plt.subplots()
qpmr.plot.roots(roots, ax=ax)
ax.plot(retarded_zeros.real, retarded_zeros.imag, 'ro', fillstyle='none')
ax.plot(neutral_zeros.real, neutral_zeros.imag, 'bo', fillstyle='none')
