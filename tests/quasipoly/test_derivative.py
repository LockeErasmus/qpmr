"""
Test analytical and numerical derivative of quasipolynomials
"""

import numpy as np
import numpy.typing as npt
import pytest

import qpmr.quasipoly.examples as examples
from qpmr.quasipoly import eval, derivative

DEFAULT_TEST_POINTS = [0+0j, 1+1j, -1-1j, 0.5+0.5j]
DEFAULT_EPS = 1e-8
DEFAULT_ABS_TOL = 1e-6
DEFAULT_REL_TOL = 1e-4

@pytest.mark.parametrize(
    argnames="qp, params",
    argvalues=[
        (examples.vyhlidal2014qpmr_01(), {}),
    ],
    ids=["vyhlidal2014qpmr_01"],
)
def test_derivatives(qp, params):
    
    coefs, delays = qp # unpack quasipolynomial
    coefs_prime, delays_prime = derivative(coefs, delays, compress=False)

    eps = params.get("eps", DEFAULT_EPS)
    abs_tol = params.get("abs_tol", DEFAULT_ABS_TOL)
    rel_tol = params.get("rel_tol", DEFAULT_REL_TOL)

    for s in params.get("points", DEFAULT_TEST_POINTS):
        analytical_derivative = eval(coefs_prime, delays_prime, s)
        
        numerical_derivative = (eval(coefs, delays, s + eps) - eval(coefs, delays, s - eps))/ 2. / eps

        assert np.isclose(analytical_derivative, numerical_derivative, atol=abs_tol, rtol=rel_tol), \
            f"Derivative mismatch at s={s}: analytical={analytical_derivative}, numerical={numerical_derivative}"




