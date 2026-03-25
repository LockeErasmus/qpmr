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
        (examples.ndiff_01(), (), {}),
        (examples.appeltans2023analysis(2.6, ), (), {}),
        (examples.appeltans2023analysis(2.6, tau2=2.05), (), {}),
        (examples.appeltans2023analysis(2.6, tau2=2.005), (), {}),
        (examples.appeltans2023analysis(2.6, tau2=2.05), (), {"region": (-10, 10, 0, 2000)}),
        (examples.vyhlidal2014qpmr_02(), (), {"region": (-6, 2, 0, 200)}),
        (examples.vyhlidal2014qpmr_03(), (), {"region": (-6, 2, 0, 1000)}),
        (examples.qp_from_roots((-1, -1, -2, -2, -3+1j, -3-1j)), (), {"region": (-5, 0, 0, 5), "multiplicity_heuristic": True}),
        (examples.mazanti2021multiplicity(), (), {"region": (-15, 30, 0, 200), "multiplicity_heuristic": True}),
        (examples.self_inverse_polynomial(center=-3), (), {"region": (-10, 10, 0, 100), "multiplicity_heuristic": True}),
    ],
    ids=[
        "vyhlidal2014qpmr-01",
        "vyhlidal2014qpmr-02",
        "vyhlidal2014qpmr-03",
        "neutral-diff-artificial-01",
        "appeltans2023analysis-2.6",
        "appeltans2023analysis-2.6-tau2=2.05",
        "appeltans2023analysis-2.6-tau2=2.005",
        "appeltans2023analysis-2.6-tau2=2.05-region",
        "vyhlidal2014qpmr-02-region",
        "vyhlidal2014qpmr-03-region",
        "polynomial-01",
        "mazanti2021multiplicity",
        "self-inverse-polynomial",
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
