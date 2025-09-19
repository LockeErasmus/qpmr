"""
Test shift operation
"""

import numpy as np
import numpy.typing as npt
import pytest

import qpmr.quasipoly.examples as examples

import qpmr.quasipoly


@pytest.mark.parametrize(
    argnames="qp, expected_qp, params",
    argvalues=[
        (
            (
                np.empty(shape=(0,0), dtype=np.float64),
                np.empty(shape=(0,), dtype=np.float64),
            ),
            None,
            {"shift": 1},
        ),
        (
            (
                np.arange(0, 10, 1, dtype=np.float64)[:, None],
                np.arange(0, 10, 1, np.float64),
            ),
            (
                (np.arange(0, 10, 1, dtype=np.float64)[:, None]) * np.exp(2*np.arange(0, 10, 1, np.float64))[:, None],
                np.arange(0, 10, 1, np.float64),
            ),
            {"shift": 2},
        ),
        (
            (
                np.array([[0., 1], [0, 1]]), # coefs
                np.array([0., 1]), # delays
            ),
            (
                np.array([[-1., 1], [-np.e, np.e]]), # coefs
                np.array([0., 1]), # delays
            ),
            {"shift": 1},
        ),
        (
            examples.vyhlidal2014qpmr(example=2),
            None,
            {"shift": -0.5},
        ),
    ],
    ids=[
        "01-artificial_empty",
        "02-artificial_constant",
        "02-artificial_linear",
        "03-vyhlidal2014qpmr_01",
    ],
)
def test_shift(qp, expected_qp, params):
    
    coefs, delays = qp # unpack quasipolynomial
    shift = params.get("shift")

    result_coefs, result_delays = qpmr.quasipoly.shift(coefs, delays, shift)
    
    if expected_qp is not None:
        expected_coefs, expected_delays = expected_qp
        assert expected_coefs.shape == result_coefs.shape
        assert expected_delays.shape == result_delays.shape
        assert np.all(expected_coefs == result_coefs)
        assert np.all(expected_delays == result_delays)

    original_coefs, original_delays = qpmr.quasipoly.shift(result_coefs, result_delays, -shift)

    assert np.allclose(coefs, original_coefs)


    