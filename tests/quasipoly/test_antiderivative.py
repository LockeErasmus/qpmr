"""
Test analytical and numerical antiderivative of quasipolynomials
"""

import numpy as np
import numpy.typing as npt
import pytest

import qpmr.quasipoly.examples as examples
from qpmr.quasipoly import eval, derivative, antiderivative


@pytest.mark.parametrize(
    argnames="qp, params",
    argvalues=[
        (examples.vyhlidal2014qpmr_01(), {}),
        (examples.vyhlidal2014qpmr_02(), {}),
        (examples.vyhlidal2014qpmr_03(), {}),
    ],
    ids=[],
)
def test_antiderivatives(qp, params):
    
    coefs, delays = qp # unpack quasipolynomial
    coefs_anti, delays_anti = antiderivative(coefs, delays, compress=False)

    print(coefs, "\n delays=", delays)

    print("\n")

    print(coefs_anti, "\n delays=", delays_anti)


    coefs, delays = derivative(coefs_anti, delays_anti, compress=True)

    print("\n")
    
    print(coefs, "\n delays=", delays)
    
    # plt.figure()
    # plt.semilogy(eps_vector, np.abs(numerical_derivative), "x-")
    # plt.semilogy(eps_vector, np.abs(analytical_derivative), "r-")
    # plt.show()






