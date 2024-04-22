"""
Set of tests quassipolynomials
"""

import numpy as np

from qpmr.quasipoly import QuasiPolynomial

def test_minimal_form_1():
    delays = np.array([0.0, 1.0, 0.0, 1.0, 2.5])
    coefs = np.array([[0, 1],
                      [1, 0],
                      [0, 2],
                      [3, 0],
                      [0, 0.]])

    minimal_delays = np.array([0., 1.])
    minimal_coefs = np.array([[0, 3],
                              [4, 0.]])
    qp = QuasiPolynomial(coefs, delays)
    qp_minimal = qp.minimal_form()

    assert qp_minimal.degree == 1
    assert qp_minimal.m == 2
    assert qp_minimal.n == 2
    assert np.all(qp_minimal.coefs == minimal_coefs)
    assert np.all(qp_minimal.delays == minimal_delays)

def test_minimal_form_empty():
    """ empty test for non-trivial qp """
    delays = np.array([1.0, 1.0, 1.0])
    coefs = np.array([[0, -1, 1.5, 0, 0, -10],
                      [0, 0.3, -0.5, 0, 0, -10],
                      [0, 0.7, -1, 0, 0, 20]])
    qp = QuasiPolynomial(coefs, delays)
    qp_minimal = qp.minimal_form()

    assert qp_minimal.is_empty

def test_minimal_form_empty_2():
    """ empty test for p(s) = 0 * exp(-s) """
    delays = np.array([1.0])
    coefs = np.array([[0.,]])
    qp = QuasiPolynomial(coefs, delays)
    qp_minimal = qp.minimal_form()

    assert qp_minimal.is_empty

# arithmetic operations

## __add__
def test_add_empty_1():
    """ empty QP + empty QP = empty QP"""
    empty_qp = QuasiPolynomial(
        np.empty(shape=(0,0), dtype=np.float64),
        np.empty(shape=(0,), dtype=np.float64),
    )
    qp_result = empty_qp + empty_qp

    assert qp_result.is_empty

def test_add_empty_2():
    empty_qp = QuasiPolynomial(
        np.empty(shape=(0,0), dtype=np.float64),
        np.empty(shape=(0,), dtype=np.float64),
    )
    qp = QuasiPolynomial(
        np.array([[0, 1],[1, 0]], dtype=np.float64),
        np.array([0.0, 1.0], dtype=np.float64),
    )

    qp_result = qp + empty_qp

    assert np.all(qp_result.coefs == qp.coefs)
    assert np.all(qp_result.delays == qp.delays)
    
def test_add_constant_1():
    const = 5.5
    qp = QuasiPolynomial(
        np.array([[0, 1],[1, 0]], dtype=np.float64),
        np.array([0.0, 1.0], dtype=np.float64),
    )

    correct_coefs = np.array([[const, 1],[1, 0]], dtype=np.float64)
    correct_delays = np.array([0.0, 1.0], dtype=np.float64)

    qp_result = (qp + const).minimal_form()

    assert np.all(qp_result.coefs == correct_coefs)
    assert np.all(qp_result.delays == correct_delays)

def test_add_constant_2():
    const = 20
    qp = QuasiPolynomial(
        np.array([[0, 1],[1, 0]], dtype=np.float64),
        np.array([0.0, 1.0], dtype=np.float64),
    )

    correct_coefs = np.array([[const, 1],[1, 0]], dtype=np.float64)
    correct_delays = np.array([0.0, 1.0], dtype=np.float64)

    qp_result = (qp + const).minimal_form()

    assert np.all(qp_result.coefs == correct_coefs)
    assert np.all(qp_result.delays == correct_delays)

def test_add_1():
    delays = np.array([0.0, 1.0])
    coefs = np.array([[0, 1],[1, 0]])
    qp = QuasiPolynomial(coefs, delays)
    qp_result = (qp + qp).minimal_form()

    assert np.all(qp_result.coefs == 2*coefs)
    assert np.all(qp_result.delays == delays)

