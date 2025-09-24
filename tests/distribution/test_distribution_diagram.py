"""
Tests implemented distribution diagram as described

TODO
"""

import pytest
import numpy as np

import qpmr.quasipoly.examples as examples
from qpmr import distribution_diagram, chain_asymptotes
from qpmr.qpmr_v3 import qpmr as qpmr_v3
import qpmr.plot

@pytest.mark.parametrize(
    argnames="qp, qpmr_args, qpmr_kwargs",
    argvalues=[
        (
            (
                np.array([
                    [1., 1, 0, 0, 0],
                    [1., 1, 1, 0, 0],
                    [1., 1, 1, 1, 0],
                    [1., 1, 0, 0, 0],
                    [1., 1, 1, 1, 1],
                    [1., 1, 1, 1, 1],
                    [1., 1, 1, 1, 0],
                    [1., 1, 1, 0, 0],
                ]),
                np.array([0., 1, 2, 4, 8, 10, 12, 14])
            ),
            (),
            {},
        ),
        (examples.vyhlidal2014qpmr_02(), (), {}),
    ],
    ids=[
        "advanced",
        "vyhlidal2014qpmr-02",
    ],
)
def test_distribution_diagram(qp, qpmr_args: tuple, qpmr_kwargs: dict, enable_plot: bool):

    import numpy as np
    
    coefs, delays = qp
    region=(-10, 20, 0, 100)

    coefs, delays = qpmr.quasipoly.compress(coefs, delays)

    x, y, mask = distribution_diagram(coefs, delays)
    mi, wk_abs = chain_asymptotes(coefs, delays)
    

    roots, meta = qpmr_v3(coefs, delays, region)

    if enable_plot:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1)
        qpmr.plot.delay_distribution.spectrum_distribution_diagram(x, y, mask, ax=ax1)
        
        qpmr.plot.roots(roots, ax=ax2)
        qpmr.plot.chain_asymptotes(mi, wk_abs, region, ax=ax2)


        plt.show()
