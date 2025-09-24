"""
These tests are comprehensive tests of a lot of different functionalities

"""

import pytest
import qpmr.quasipoly.examples as examples
from qpmr.qpmr_v3 import qpmr as qpmr_v3
from qpmr.qpmr_v3 import QpmrSubInfo, QpmrRecursionContext

from qpmr.distribution.envelope_curve import _spectral_norms

import qpmr.plot

@pytest.mark.parametrize(
    argnames="qp, region, qpmr_args, qpmr_kwargs",
    argvalues=[
        (examples.appeltans2023analysis(example="2.6", tau2=2.005), (-1, 2.5, -100, 500), (), {}),
    ],

    ids=[
        "appeltans2023analysis-2.6-tau2=2.05",
    ],
)
def test_neutral(qp, region, qpmr_args: tuple, qpmr_kwargs: dict, enable_plot: bool):

    coefs, delays = qp

    print(coefs, delays)
    
    roots, ctx = qpmr_v3(coefs, delays, region, **qpmr_kwargs)

    norms = _spectral_norms(coefs, delays)

    if enable_plot:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1)
        
        qpmr.plot.qpmr_solution_tree(ctx, ax=ax1)

        qpmr.plot.spectrum_envelope(norms, delays, region, ax=ax2)
        qpmr.plot.roots(roots, ax=ax2)
        plt.show()
