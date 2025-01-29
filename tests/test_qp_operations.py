"""
Set of tests for usefull operations on quasipolynomials
"""

import numpy as np

from qpmr.quasipoly import compress_qp

def test_compress_1():
    delays = np.array([0.0, 1.0, 0.0, 1.0, 2.5])
    coefs = np.array([[0, 1],
                      [1, 0],
                      [0, 2],
                      [3, 0],
                      [0, 0.]])

    expected_delays = np.array([0., 1.])
    expected_coefs = np.array([[0, 3],
                               [4, 0.]])
    
    result_coefs, result_delays = compress_qp(coefs, delays)

    assert expected_coefs.shape == result_coefs.shape
    assert expected_delays.shape == result_delays.shape
    assert np.all(expected_coefs == result_coefs)
    assert np.all(expected_delays == result_delays)

def test_compress_2_empty():
    """ empty test for non-trivial qp """
    delays = np.array([1.0, 1.0, 1.0])
    coefs = np.array([[0, -1, 1.5, 0, 0, -10],
                      [0, 0.3, -0.5, 0, 0, -10],
                      [0, 0.7, -1, 0, 0, 20]])

    expected_delays = np.array([])
    expected_coefs = np.array([[]])
    
    result_coefs, result_delays = compress_qp(coefs, delays)

    assert expected_coefs.shape == result_coefs.shape
    assert expected_delays.shape == result_delays.shape
    assert np.all(expected_coefs == result_coefs)
    assert np.all(expected_delays == result_delays)

def test_minimal_form_empty_2():
    """ empty test for p(s) = 0 * exp(-s) """
    delays = np.array([1.0])
    coefs = np.array([[0.,]])

    expected_delays = np.array([])
    expected_coefs = np.array([[]])
    
    result_coefs, result_delays = compress_qp(coefs, delays)

    assert expected_coefs.shape == result_coefs.shape
    assert expected_delays.shape == result_delays.shape
    assert np.all(expected_coefs == result_coefs)
    assert np.all(expected_delays == result_delays)