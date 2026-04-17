"""
Script for MATLAB performance comparison for TDS 2024 workshop presentation
---------------------------------------------------------------------------
"""
import logging

import time
import numpy as np

from qpmr import qpmr
import qpmr.quasipoly.examples as examples


tests = [
    {   
        "qp": examples.vyhlidal2014qpmr(1),
        "settings": [
            {"region": (-10, 2, 0, 30), "N": 10_000, "N_roots_expected": 5},
            {"region": (-20, 2, 0, 200), "N": 10_000, "N_roots_expected": 32},
            {"region": (-40, 2, 0, 1000), "N": 1_000, "N_roots_expected": 159},
        ],
    },
    {   
        "qp": examples.vyhlidal2014qpmr(2),
        "settings": [
            {"region": (-2.25, 3, 0, 20), "N": 100, "N_roots_expected": 83},
            {"region": (-4.5, 3, 0, 100), "N": 50, "N_roots_expected": 401},
            {"region": (-9, 3, 0, 200), "N": 50, "N_roots_expected": 798},
        ],
    },
    {   
        "qp": examples.vyhlidal2014qpmr(3),
        "settings": [
            {"region": (-2, 7, 0, 9), "N": 10_000, "N_roots_expected": 10},
            {"region": (-10, 7, 0, 100), "N": 1_000, "N_roots_expected": 101},
            {"region": (-20, 20, 0, 200), "N": 1_000, "N_roots_expected": 201},
        ],
    },
]

if __name__ == "__main__":
    logger = logging.getLogger("qpmr")
    logging.basicConfig(level=logging.WARNING)
    
    for test in tests:
        coefs, delays = test["qp"]

        print(f"Testing QP with coefs=\n{coefs} and delays=\n{delays}")
        print(80*"=")
        
        for setting in test["settings"]:
            region = setting["region"]
            N = setting["N"]
            N_roots_expected = setting["N_roots_expected"]
    
            time_vector = np.full(shape=(N,), fill_value=np.nan)
            
            for i in range(N):
                s = time.time()
                roots, meta = qpmr(coefs, delays, region=region)
                time_vector[i] = time.time() - s
            
            print(f"    Test finished for {region=}, | {np.round(np.mean(time_vector)*1000, 0)} $\pm$ "
                  f"{np.round(np.std(time_vector)*1000, 0)} ms | {N=} | Nroots={len(roots)} (expected: {N_roots_expected})")
