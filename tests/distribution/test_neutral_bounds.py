"""
Tests implemented distribution diagram as described

TODO
"""

import pytest
import numpy as np

import qpmr.quasipoly.examples as examples
from qpmr import bounds_neutral_strip, distribution_diagram
from qpmr.qpmr_v3 import qpmr as qpmr_v3
import qpmr.plot

@pytest.mark.parametrize(
    argnames="qp, qpmr_args, qpmr_kwargs",
    argvalues=[
        (
            (
                np.array([
                    [0, 1.0000],
                    [1.0000, 1.4142],
                    [1.0000, 1.0000],
                    [1.0000, 2.2361],
                    [0, 8.0987],
                ]),
                np.array([0., 1., np.sqrt(2), 1.6, np.sqrt(5)])
            ),
            (),
            {"region": None},
        ),
    ],
    ids=[
        "artificial02",
    ],
)
def test_neutral_strip(qp, qpmr_args: tuple, qpmr_kwargs: dict, enable_plot: bool):

    import numpy as np
    
    coefs, delays = qp
    cdm, cdp = qpmr.bounds_neutral_strip(coefs, delays)    

    if enable_plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        roots, meta = qpmr_v3(coefs, delays)
        qpmr.plot.roots(roots, ax=ax)
        ax.axvline(cdm, color="blue")
        ax.axvline(cdp, color="blue")

        plt.show()
