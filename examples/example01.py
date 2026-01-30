"""
Example 1
=========

We start with he introductory quasi-polynomial

.. math::

    h(s) = e




"""

import numpy as np
import matplotlib.pyplot as plt
import qpmr
import qpmr.plot


delays = np.array([0.0, 1.0])
coefs = np.array([[0., 1],[1, 0]])

roots, meta = qpmr.qpmr(coefs, delays, region=[-10, 2, 0, 30])

qpmr.plot.roots(roots)
plt.show()
