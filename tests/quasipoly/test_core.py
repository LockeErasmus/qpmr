"""
Test of core operations
"""

import numpy as np
import numpy.typing as npt
import pytest

import qpmr.quasipoly.examples as examples
from qpmr.quasipoly.core import normalize


@pytest.mark.parametrize(
    argnames="qp, expected_qp, params",
    argvalues=[
        (
            (
                np.vstack([np.arange(1, 5, dtype=float) for _ in range(20)]), # coefs
                np.hstack([np.full(10, -1.), np.full(10, 1.)]), # delays
            ),
            (
                1/40. * np.array([[10., 20, 30, 40],[10., 20, 30, 40]]),
                np.array([0, 2.]),
            ),
            {},
        ),
    ],
    ids=[
        "01-artificial",
    ],
)
def test_normalize(qp, expected_qp, params):
    
    coefs, delays = qp # unpack quasipolynomial
    expected_coefs, expected_delays = expected_qp

    result_coefs, result_delays = normalize(coefs, delays)
    assert expected_coefs.shape == result_coefs.shape
    assert expected_delays.shape == result_delays.shape
    assert np.all(expected_coefs == result_coefs)
    assert np.all(expected_delays == result_delays)
    