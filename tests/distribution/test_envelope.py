"""
Test functionality connected to spectrum envelope calculation
"""

import numpy as np
import numpy.typing as npt
import pytest

import qpmr.quasipoly.examples as examples
from qpmr.quasipoly import eval, derivative, compress

from qpmr.distribution.envelope_curve import _envelope_real_axis_crossing, _spectral_norms

@pytest.mark.parametrize(
    argnames="qp, expected, params",
    argvalues=[
        (examples.vyhlidal2014qpmr_02(), 2.75, {}),
    ],
    ids=[
        "vyhlidal2014qpmr-02",
    ],
)
def test_envelope_real_axis_crossing(qp, expected, params):
    
    plot = True
    
    coefs, delays = qp # unpack quasipolynomial
    coefs, delays = compress(coefs, delays)
    coefs /= coefs[0, -1] # normalize
    norms = _spectral_norms(coefs, delays)
    x_star = _envelope_real_axis_crossing(norms, delays)

    assert abs(x_star - expected) < 0.1


    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(eps_vector, np.abs(numerical_derivative), "x-")
    # plt.plot(eps_vector, np.abs(analytical_derivative), "r-")
    # plt.show()

    # fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)

    # ax1.plot(eps_vector, np.abs(numerical_derivative.real - analytical_derivative.real))
    # ax2.plot(eps_vector, np.abs(numerical_derivative.imag - analytical_derivative.imag))
    # plt.show()




