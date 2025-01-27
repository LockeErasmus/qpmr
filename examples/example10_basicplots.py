"""
Example 10 - qpmr.plot
"""
import logging

import numpy as np

region = [-10, 2, 0, 300]
delays = np.array([0.0, 1.0])
coefs = np.array([[0, 1],[1, 0]])

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    import qpmr
    import qpmr.utils
    import qpmr.plot

    qpmr.utils.init_qpmr_logger(level="DEBUG")
    
    # logging.getLogger("matplotlib").setLevel(logging.ERROR)
    # logging.getLogger("PIL").setLevel(logging.ERROR)
    # logger = logging.getLogger("qpmr")
    # logging.basicConfig(level=logging.DEBUG)

    roots, meta = qpmr.qpmr(region, coefs, delays)

    qpmr.plot.roots_basic(roots)
    qpmr.plot.qpmr_basic(roots,  meta)
    
    plt.show()