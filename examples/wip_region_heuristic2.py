"""
WIP - Region heuristic 2
"""

import qpmr.quasipoly.examples as examples
from qpmr.region_heuristic import region_heuristic
from qpmr.qpmr_v3 import qpmr as qpmr_v3
from qpmr.qpmr_v3 import QpmrSubInfo, QpmrRecursionContext
from qpmr.distribution.zero_chains import chain_asymptotes, chain_asymptotes2
from qpmr.argument_principle import argument_principle_rectangle
import qpmr.quasipoly
import matplotlib.pyplot as plt

from qpmr.distribution.envelope_curve import _spectral_norms, _envelope_real_axis_crossing, _envelope_eval
import qpmr.plot

if __name__ == "__main__":
    
    qpmr.init_logger(level="DEBUG")

    import numpy as np
    
    coefs, delays = examples.vyhlidal2014qpmr(example=2)

    # coefs, delays = examples.mazanti2021multiplicity()

    coefs, delays = qpmr.quasipoly.compress(coefs, delays)

    # mi_vec, abs_omega = chain_asymptotes(coefs, delays)
    norms = _spectral_norms(coefs, delays)
    re_max = _envelope_real_axis_crossing(norms, delays)


    # calculation
    tau_max = delays[-1]
    im_max = np.pi/tau_max * 2000 # Tomas wants 50 roots

    # obtain re_min
    mu, w = chain_asymptotes(coefs, delays)

    points = []
    for i in range(len(mu)):
        for ww in w[i]:
            # calculate intersection with im_max
            points.append(
                -mu[i] * np.log(im_max/ww)
            )

    re_min = min(points) # - 2*np.pi/tau_max
    region = (re_min, re_max, 0, im_max)

    roots, ctx = qpmr_v3(coefs, delays, region)

    print(f"Total number of roots found: {len(roots)}")

    fig, ax = plt.subplots(1,1)
    qpmr.plot.roots(roots, ax=ax)

    qpmr.plot.chain_asymptotes(mu, w, region=region, ax=ax)

    ax.axhline(im_max)
    ax.axvline(re_min)
    ax.scatter(points, [re_min for _ in points], color="b")

    fig, ax = plt.subplots(1,1)
    qpmr.plot.qpmr_solution_tree(ctx, ax=ax)

    
    plt.show()
