"""
Set of tests for novel region heuristic
---------------------------------------

"""
import pytest

import qpmr.quasipoly.examples as examples
from qpmr.region_heuristic import region_heuristic
from qpmr.qpmr_v3 import qpmr as qpmr_v3
from qpmr.qpmr_v3 import QpmrSubInfo, QpmrRecursionContext
from qpmr.distribution.zero_chains import chain_asymptotes
from qpmr.argument_principle import argument_principle_rectangle
import qpmr.quasipoly

from qpmr.distribution.envelope_curve import _spectral_norms, _envelope_real_axis_crossing, _envelope_eval
import qpmr.plot

@pytest.mark.parametrize(
    argnames="qp, qpmr_args, qpmr_kwargs",
    argvalues=[
        (examples.vyhlidal2014qpmr_02(), (), {}),
    ],
    ids=[
        "vyhlidal2014qpmr-02",
    ],
)
def test_region_heuristic(qp, qpmr_args: tuple, qpmr_kwargs: dict, enable_plot: bool):

    import numpy as np
    
    coefs, delays = qp
    region=(-10, 10, 0, 100)

    coefs, delays = qpmr.quasipoly.compress(coefs, delays)

    mi_vec, abs_omega = chain_asymptotes(coefs, delays)
    norms = _spectral_norms(coefs, delays)

    

    re_max = _envelope_real_axis_crossing(norms, delays)
    for i in range(1, 20):
        
        r = _envelope_eval(np.array([re_max - i]), norms, delays)
        region = (re_max - i, re_max, -1e-3, r[0])
        n = argument_principle_rectangle(lambda x: qpmr.quasipoly.eval(coefs, delays, x), region=region, ds=0.1, eps=1e-6)
        print(f"{region}= | {n=}")

    region_heuristic(coefs, delays, n=30)

    if enable_plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        qpmr.plot.chain_asymptotes(mi_vec, abs_omega, region=region, ax=ax)
        qpmr.plot.spectrum_envelope(norms, delays, region=region, ax=ax)
                
        # qpmr.plot.qpmr_solution_tree(ctx, ax=ax)
        plt.show()
