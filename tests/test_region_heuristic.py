"""
Set of tests for region heuristic
---------------------------------

"""
import pytest
import numpy as np

import qpmr.quasipoly.examples as examples
from qpmr.region_heuristic import region_heuristic

@pytest.mark.parametrize(
    argnames="qp, args, kwargs",
    argvalues=[
        (examples.vyhlidal2014qpmr_01(), (), {}),
        (examples.vyhlidal2014qpmr_02(), (), {}),
        (examples.vyhlidal2014qpmr_03(), (), {}),
        (examples.ndiff_01(), (), {"n_roots": 700}),
    ],
    ids=[
        "vyhlidal2014qpmr-01",
        "vyhlidal2014qpmr-02",
        "vyhlidal2014qpmr-03",
        "ndiff-01",
    ],
)
def test_region_heuristic(qp, args: tuple, kwargs: dict, enable_plot: bool):
    coefs, delays = qp
    region = region_heuristic(coefs, delays, *args, **kwargs)

    assert isinstance(region, tuple)
    assert len(region) == 4
    assert all(isinstance(x, float) for x in region)
    assert region[1] > region[0]
    assert region[3] > region[2]
    assert region[2] == 0.0  # region heuristic always returns a region with lower bound of imaginary part equal to zero

    print(region)
    if enable_plot:
        import matplotlib.pyplot as plt
        import qpmr.plot
        fig, ax = plt.subplots()
        # TODO
        plt.show()
    

