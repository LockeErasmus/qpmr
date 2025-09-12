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
    argnames="qp, region, qpmr_args, qpmr_kwargs",
    argvalues=[
        (examples.vyhlidal2014qpmr_01(), (-10, 2, 0, 30), (), {}),
        (examples.vyhlidal2014qpmr_02(), (-4.5, 2.5, 0, 50), (), {}),
        (examples.vyhlidal2014qpmr_03(), (-6, 2, 0, 200), (), {}),
        (examples.mazanti2021multiplicity(), (-15, 30, 0, 200), (), {"multiplicity_heuristic": True, "recursion_level_max": 5}),
        (examples.mazanti2021multiplicity(), (-20, 100, 0, 200), (), {"multiplicity_heuristic": True, "recursion_level_max": 8}),
    ],
    ids=[
        "vyhlidal2014qpmr-01",
        "vyhlidal2014qpmr-02",
        "vyhlidal2014qpmr-03",
        "mazanti2021multiplicity",
        "test",
    ],
)
def test_qpmr_v3(qp, region, qpmr_args: tuple, qpmr_kwargs: dict, enable_plot: bool):

    coefs, delays = qp
    
    roots, ctx = qpmr_v3(coefs, delays, region, **qpmr_kwargs)

    if enable_plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        qpmr.plot.qpmr_solution_tree(ctx, ax=ax)
        plt.show()
