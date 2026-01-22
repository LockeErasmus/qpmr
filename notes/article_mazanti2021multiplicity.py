"""
Docstring for examples.article_mazanti2021multiplicity
"""

import matplotlib.pyplot as plt
import numpy as np
import qpmr
import qpmr.plot

coefs, delays = qpmr.examples.mazanti2021multiplicity()

roots, meta = qpmr.qpmr(coefs, delays, None)

qpmr.plot.qpmr_solution_tree(roots, meta)
plt.show()


