"""
Set of tests quassipolynomials
"""

import numpy as np

from qpmr.quassipoly import QuassiPolynomial

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
    qp = QuassiPolynomial(coefs, delays)
    qp_minimal = qp.minimal_form()

    assert qp_minimal.degree == 2
    assert qp_minimal.n == 2
    assert np.all(qp_minimal.coefs == minimal_coefs)
    assert np.all(qp_minimal.delays == minimal_delays)

def test_add():
    pass
    