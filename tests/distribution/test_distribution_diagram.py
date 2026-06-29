"""
Tests implemented distribution diagram as described

TODO
"""

import pytest
import numpy as np

import qpmr
import qpmr.quasipoly.examples as examples
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
            {"region": (-6, 10, 0, 50)},
        ),
        (
            (
                np.array([
                    [1., 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1., 1, 0, 0, 0, 0, 0, 0, 1, 0.3, 0],
                    [1., 1, 0, 0, 0, 0, 0, 1, 0.8, 0, 0],
                    [1., 1, 0, 0, 0, 0, 1, 2.1, 0, 0, 0],
                    [1., 1, 0, 0, 0, 0, 4.6, 0, 0, 0, 0],
                    [1., 1, 0, 0, 0, 0.12, 0, 0, 0, 0, 0],
                ]),
                np.array([0., 0.6 , 1.7, 2.8, 3.9, 5])
            ),
            (),
            {},
        ),
        (examples.vyhlidal2014qpmr_02(), (), {"region": (-6, 2, 0, 100)}),
    ],
    ids=[
        "advanced",
        "artificial02",
        "vyhlidal2014qpmr-02",
    ],
)
def test_distribution_diagram(qp, qpmr_args: tuple, qpmr_kwargs: dict, enable_plot: bool):
    import numpy as np
    
    coefs, delays = qp
    x, y, mask = qpmr.distribution_diagram(coefs, delays)
    
    if enable_plot:
        import matplotlib.pyplot as plt

        roots, info = qpmr.qpmr(coefs, delays, **qpmr_kwargs)
        mi, wk_abs = qpmr.chain_asymptotes(coefs, delays)
        
        fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1)
        qpmr.plot.spectrum_distribution_diagram(x, y, mask, ax=ax1)
        
        qpmr.plot.roots(roots, ax=ax2)
        qpmr.plot.chain_asymptotes(mi, wk_abs, region=info.region, ax=ax2)

        plt.show()
