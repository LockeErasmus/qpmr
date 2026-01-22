"""
Script for MATLAB performance comparison for TDS 2024 workshop presentation
---------------------------------------------------------------------------
"""
import logging

import time
import numpy as np

from qpmr import qpmr

if __name__ == "__main__":
    logger = logging.getLogger("qpmr")
    logging.basicConfig(level=logging.WARNING)
    
    # SELECT one of the regions
    #region = [-2, 7, 0, 9]
    #region = [-10, 7, 0, 100]
    region = [-10, 7, 0, 200]

    # QP definition
    delays = np.array([0.0, 1.3, 3.5, 4.3])
    coefs = np.array([[20.1, 0, 0.2, 1.5],
                      [0, -2.1, 0, 1],
                      [0, 3.2, 0, 0],
                      [1.4, 0, 0, 0]])
    
    N = 1000 # number of runs
    
    
    time_vector = np.full(shape=(N,), fill_value=np.nan)
    for i in range(N):

        s = time.time()
        roots, meta = qpmr(region, coefs, delays)
        time_vector[i] = time.time() - s
    
    logger.warning((f"Test finished for {region=}, | {np.mean(time_vector)} +- "
                    f"{np.std(time_vector)} s | {N=}"))
