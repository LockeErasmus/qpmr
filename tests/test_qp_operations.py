"""
Set of tests for usefull operations on quasipolynomials
"""

import numpy as np

import pytest

from qpmr.quasipoly import compress
from qpmr.quasipoly.arithmetic import add, multiply

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
    
    result_coefs, result_delays = compress(coefs, delays)

    assert expected_coefs.shape == result_coefs.shape
    assert expected_delays.shape == result_delays.shape
    assert np.all(expected_coefs == result_coefs)
    assert np.all(expected_delays == result_delays)

def test_compress_empty_1():
    """ empty test for non-trivial qp """
    delays = np.array([1.0, 1.0, 1.0])
    coefs = np.array([[0, -1, 1.5, 0, 0, -10],
                      [0, 0.3, -0.5, 0, 0, -10],
                      [0, 0.7, -1, 0, 0, 20]])

    expected_delays = np.array([])
    expected_coefs = np.array([[]])
    
    result_coefs, result_delays = compress(coefs, delays)

    assert expected_coefs.shape == result_coefs.shape
    assert expected_delays.shape == result_delays.shape
    assert np.all(expected_coefs == result_coefs)
    assert np.all(expected_delays == result_delays)

def test_compress_empty_2():
    """ empty test for non-trivial qp """
    delays = np.array([])
    coefs = np.array([[]])

    expected_delays = np.array([])
    expected_coefs = np.array([[]])
    
    result_coefs, result_delays = compress(coefs, delays)

    assert expected_coefs.shape == result_coefs.shape
    assert expected_delays.shape == result_delays.shape
    assert np.all(expected_coefs == result_coefs)
    assert np.all(expected_delays == result_delays)

def test_compress_empty_3():
    """ empty test for non-trivial qp """
    delays = np.array([1.0])
    coefs = np.array([[0.,]])

    expected_delays = np.array([])
    expected_coefs = np.array([[]])
    
    result_coefs, result_delays = compress(coefs, delays)

    assert expected_coefs.shape == result_coefs.shape
    assert expected_delays.shape == result_delays.shape
    assert np.all(expected_coefs == result_coefs)
    assert np.all(expected_delays == result_delays)


# adding polynomials
def test_add_1():
    coefs1 = np.array([[0, 1],[1, 0]], dtype=np.float64)
    delays1 = np.array([0.0, 1.0])
    coefs2 = coefs1
    delays2 = delays1

    expected_coefs = 2*coefs1
    expected_delays = delays1

    result_coefs, result_delays = add(coefs1, delays1, coefs2, delays2)

    assert expected_coefs.shape == result_coefs.shape
    assert expected_delays.shape == result_delays.shape
    assert np.all(expected_coefs == result_coefs)
    assert np.all(expected_delays == result_delays)

def test_add_constant_1():
    const = 2.5
    coefs1 = np.array([[0, 1],[1, 0]], dtype=np.float64)
    delays1 = np.array([0.0, 1.0], dtype=np.float64)
    coefs2 = np.array([[const]], dtype=np.float64)
    delays2 = np.array([0.0], dtype=np.float64)

    expected_coefs = np.array([[const, 1],[1, 0]], dtype=np.float64)
    expected_delays = np.array([0.0, 1.0], dtype=np.float64)

    result_coefs, result_delays = add(coefs1, delays1, coefs2, delays2)

    assert expected_coefs.shape == result_coefs.shape
    assert expected_delays.shape == result_delays.shape
    assert np.all(expected_coefs == result_coefs)
    assert np.all(expected_delays == result_delays)

def test_add_empty_1():
    """ empty QP + empty QP = empty QP"""
    delays1 = np.array([])
    coefs1 = np.array([[]])
    delays2 = np.array([])
    coefs2 = np.array([[]])

    expected_delays = np.array([])
    expected_coefs = np.array([[]])

    result_coefs, result_delays = add(coefs1, delays1, coefs2, delays2)

    assert expected_coefs.shape == result_coefs.shape
    assert expected_delays.shape == result_delays.shape
    assert np.all(expected_coefs == result_coefs)
    assert np.all(expected_delays == result_delays)

# test multiply
def test_multiply_1():
    coefs1 = np.array([[0, 1],[1, 0]], dtype=np.float64)
    delays1 = np.array([0.0, 1.0])
    coefs2 = coefs1
    delays2 = delays1

    expected_coefs = np.array([
        [0, 0, 1],
        [0, 2, 0],
        [1, 0, 0],
    ], dtype=np.float64)
    expected_delays = np.array([0, 1, 2], dtype=np.float64)

    result_coefs, result_delays = multiply(coefs1, delays1, coefs2, delays2)

    assert expected_coefs.shape == result_coefs.shape
    assert expected_delays.shape == result_delays.shape
    assert np.all(expected_coefs == result_coefs)
    assert np.all(expected_delays == result_delays)
