"""
Set of tests for spectrum mapping algorithm
-------------------------------------------

"""
import logging
import pytest

from qpmr.core.spectrum_mapping import spectrum_mapping
import qpmr.quasipoly.examples as examples
import qpmr.plot

@pytest.mark.parametrize(
    argnames="qp, args, kwargs",
    argvalues=[
        # (examples.vyhlidal2014qpmr_01(), None, (), {}),
        # (examples.vyhlidal2014qpmr_02(), None, (), {}),
        (examples.vyhlidal2014qpmr_03(), ( (-6, 2, 0, 200), 1), {}),
        # (examples.mazanti2021multiplicity(), (-15, 30, 0, 200), (), {"multiplicity_heuristic": True, "recursion_level_max": 5}),
        # (examples.mazanti2021multiplicity(), (-20, 100, 0, 200), (), {"multiplicity_heuristic": True, "recursion_level_max": 8}),
    ],
    ids=[
        # "vyhlidal2014qpmr-01",
        # "vyhlidal2014qpmr-02",
        "vyhlidal2014qpmr-03",
        # "mazanti2021multiplicity",
        # "test",
    ],
)
def test_qpmr_v3(qp, args, kwargs, enable_plot: bool):

    coefs, delays = qp
    
    roots = spectrum_mapping(coefs, delays, *args, **kwargs)

    if enable_plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        qpmr.plot.roots(roots, ax=ax, label="roots")
        plt.show()
