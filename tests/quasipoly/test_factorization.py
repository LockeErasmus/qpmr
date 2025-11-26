"""
Test shift operation
"""

import numpy as np
import numpy.typing as npt
import pytest

import qpmr.quasipoly.examples as examples

import qpmr.quasipoly


@pytest.mark.parametrize(
    argnames="qp, expected_result, params",
    argvalues=[
        (
            (
                np.empty(shape=(0,0), dtype=np.float64),
                np.empty(shape=(0,), dtype=np.float64),
            ),
            (
                np.empty(shape=(0,0), dtype=np.float64),
                np.empty(shape=(0,), dtype=np.float64),
                0,
            ),
            {},
        ),
        (
            (
                np.zeros(shape=(10,5), dtype=np.float64),
                np.arange(0, 5, 1, dtype=np.float64)
            ),
            (
                np.empty(shape=(0,0), dtype=np.float64),
                np.empty(shape=(0,), dtype=np.float64),
                0,
            ),
            {},
        ),
        (
            (
                np.c_[np.zeros(shape=(10,5), dtype=np.float64), np.ones(shape=(10, 3))],
                np.arange(0, 10, 1, dtype=np.float64)
            ),
            (
                np.ones(shape=(10, 3), dtype=np.float64),
                np.arange(0, 10, 1, dtype=np.float64),
                5,
            ),
            {},
        ),
        (
            examples.vyhlidal2014qpmr(example=2),
            (*examples.vyhlidal2014qpmr(example=2), 0),
            {},
        ),
    ],
    ids=[
        "01-artificial_empty",
        "02-artificial_zero",
        "03-artifical_ok",
        "03-vyhlidal2014qpmr_01",
    ],
)
def test_factorization(qp, expected_result, params):
    
    coefs, delays = qp # unpack quasipolynomial
    result_coefs, result_delays, result_power = qpmr.quasipoly.factorize_power(coefs, delays)
    
    if expected_result is not None:
        expected_coefs, expected_delays, expected_power = expected_result
        assert expected_power == result_power
        assert expected_coefs.shape == result_coefs.shape
        assert expected_delays.shape == result_delays.shape
        assert np.all(expected_coefs == result_coefs)
        assert np.all(expected_delays == result_delays)


    