"""
Test functionality connected to spectrum envelope calculation
"""

import numpy as np
import numpy.typing as npt
import pytest

import qpmr.quasipoly.examples as examples
from qpmr.quasipoly import eval, derivative, compress
from qpmr.distribution.envelope_curve import _spectral_norms, _envelope_real_axis_crossing, _envelope_imag_axis_crossing

@pytest.mark.parametrize(
    argnames="qp, expected, params",
    argvalues=[
        (examples.vyhlidal2014qpmr_01(), None, {}),
        (examples.vyhlidal2014qpmr_02(), None, {}),
        (examples.vyhlidal2014qpmr_03(), None, {}),
        (examples.appeltans2023analysis(example="2.6", tau2=2.0), None, {}),
    ],
    ids=[
        "vyhlidal2014qpmr-01",
        "vyhlidal2014qpmr-02",
        "vyhlidal2014qpmr-03",
        "appeltans2023analysis-2.6",
    ],
)
def test_envelope_real_axis_crossing(qp, expected: float, params: dict, enable_plot: bool):    
    
    coefs, delays = qp # unpack quasipolynomial
    coefs, delays = compress(coefs, delays)
    coefs /= coefs[0, -1] # normalize
    norms = _spectral_norms(coefs, delays)
    re_star = _envelope_real_axis_crossing(norms, delays)
    im_star = _envelope_imag_axis_crossing(norms)

    if expected is not None:
        assert abs(re_star - expected) < params.get("abs_tol", 0.1)

    if enable_plot:
        import matplotlib.pyplot as plt
        import contourpy

        fig, ax = plt.subplots()
        
        real_range = np.arange(-2, re_star+5, 0.01)
        imag_range = np.arange(-im_star*1.1+1, im_star-1.1+1, 0.01)
        complex_grid = 1j*imag_range.reshape(-1, 1) + real_range
        r = np.sum(np.exp(-np.real(complex_grid)[:,:,None]*delays[None,:])*norms, axis=-1) - np.abs(complex_grid)
        contour_generator = contourpy.contour_generator(x=real_range, y=imag_range, z=r)
        zero_level_contours = contour_generator.lines(0.0) # find all 0-level real contours

        ax.axhline(0, alpha=0.5, color="black", linestyle="-.")
        ax.axvline(0, alpha=0.5, color="black", linestyle="-.")

        for i, c in enumerate(zero_level_contours): # there should be only one, but plot all
            ax.plot(c[:,0], c[:,1], color="b", alpha=0.5, label=f"envelope (0-contour ix:{i})")
        ax.axvline(re_star, color="r", alpha=0.5, label="re*")
        ax.plot([0,0], [-im_star, im_star], marker="o", color="r", alpha=0.5, label="im*")

        ax.legend()
        plt.show()




