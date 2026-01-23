"""
Set of tests for region heuristic
---------------------------------

"""
import pytest

import qpmr
import qpmr.quasipoly.examples as examples
from qpmr.region_heuristic import region_heuristic
import qpmr.plot

@pytest.mark.parametrize(
    argnames="qp, args, kwargs",
    argvalues=[
        (examples.vyhlidal2014qpmr_01(), (), {}),
        (examples.vyhlidal2014qpmr_03(), (), {}),
        (examples.ndiff_01(), (), {"n_roots": 700}),
    ],
    ids=[
        "vyhlidal2014qpmr-01",
        "vyhlidal2014qpmr-03",
        "ndiff-01",
    ],
)
def test_region_heuristic(qp, args: tuple, kwargs: dict, enable_plot: bool):
    
    coefs, delays = qp

    region = region_heuristic(coefs, delays, *args, **kwargs)
    
    print(region)
    
    roots, info = qpmr.qpmr(coefs, delays, region=region)
    
    if enable_plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        qpmr.plot.qpmr_solution_tree(info, ax=ax)
        plt.show()
    

