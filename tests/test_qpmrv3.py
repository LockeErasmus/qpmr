"""
Set of tests for qpmr_v3 algorithm
----------------------------------

"""
import logging
import pytest
import qpmr.quasipoly.examples as examples
from qpmr.qpmr_v3 import qpmr as qpmr_v3
from qpmr.qpmr_v3 import QpmrSubInfo, QpmrRecursionContext
import qpmr.plot

@pytest.mark.parametrize(
    argnames="qp, args, kwargs",
    argvalues=[
        (examples.vyhlidal2014qpmr_01(), (), {}),
        (examples.vyhlidal2014qpmr_02(), (), {}),
        (examples.vyhlidal2014qpmr_03(), (), {}),
        (examples.vyhlidal2014qpmr_02(), (), {"region": (-6, 2, 0, 200)}),
        (examples.vyhlidal2014qpmr_03(), (), {"region": (-6, 2, 0, 1000)}),
        (examples.mazanti2021multiplicity(), (), {"region": (-15, 30, 0, 200), "multiplicity_heuristic": True, "recursion_level_max": 5}),
        (examples.mazanti2021multiplicity(), (), {"region": (-20, 100, 0, 200), "multiplicity_heuristic": True, "recursion_level_max": 8}),
    ],
    ids=[
        "vyhlidal2014qpmr-01",
        "vyhlidal2014qpmr-02",
        "vyhlidal2014qpmr-03",
        "vyhlidal2014qpmr-02-region",
        "vyhlidal2014qpmr-03-region",
        "mazanti2021multiplicity",
        "test",
    ],
)
def test_qpmr_v3(qp, args, kwargs, enable_plot: bool):

    coefs, delays = qp
    
    roots, ctx = qpmr_v3(coefs, delays, *args, **kwargs)

    if enable_plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        qpmr.plot.roots(roots, ax=ax)
        plt.show()
