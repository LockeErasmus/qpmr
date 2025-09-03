"""
Example 11
"""

import numpy as np

import qpmr
from qpmr.quasipoly import multiply
import qpmr.plot

c = 3
degree = 6
scale = 10.
coefs = ( (scale * np.poly1d([1, -c])) ** degree - 1 ).coeffs[::-1]
coefs1 = coefs[None, :]
delays1 = np.array([0.])

coefs = np.array([[1,0],[0,1.]])
delays = np.array([0, 1.])
coefs2, delays2 = multiply(coefs, delays, coefs, delays)
coefs2, delays2 = multiply(coefs2, delays2, coefs, delays)


coefs, delays = multiply(coefs1, delays1, coefs2, delays2)

region = (0, 10, -0.1, 50)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import qpmr
    import qpmr.plot

    from qpmr.qpmr_v3 import qpmr as qpmr_v3

    logger = qpmr.init_logger(level="DEBUG", format="%(name)s - %(message)s")

    roots, meta = qpmr_v3(region, coefs, delays, numerical_method_kwargs={"max_iterations": 1000, "tolerance": 1e-5}, multiplicity_heuristic=True)

    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(8,5))

    qpmr.plot.qpmr_solution_tree(meta, ax=ax1)
    qpmr.plot.roots(roots, ax=ax2)

    plt.show()
