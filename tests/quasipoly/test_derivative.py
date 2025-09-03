"""
Test analytical and numerical derivative of quasipolynomials
"""

import numpy as np
import numpy.typing as npt
import pytest

import qpmr.quasipoly.examples as examples
from qpmr.quasipoly import eval, derivative


@pytest.mark.parametrize(
    argnames="qp, params",
    argvalues=[
        (examples.vyhlidal2014qpmr_01(), {}),
    ],
    ids=[],
)
def test_derivatives(qp, params):
    
    coefs, delays = qp # unpack quasipolynomial
    coefs_prime, delays_prime = derivative(coefs, delays, compress=False)

    print(coefs, delays)
    print(coefs_prime, delays_prime)

    s = 0.5+0.23j
    numerical_derivative = []
    eps_vector = np.logspace(-3, 0, 20)
    for eps in eps_vector:
        ceps = eps * (1+1j)
        f1 = (eval(coefs, delays, s + ceps) - eval(coefs, delays, s - ceps))/ 2. / ceps
        f2 = (eval(coefs, delays, s + eps) - eval(coefs, delays, s - eps) -1j*(eval(coefs, delays, s + 1j*eps) - eval(coefs, delays, s - 1j*eps)) ) / 4. / eps
        numerical_derivative.append(
            f2
        )
    
    numerical_derivative = np.array(numerical_derivative)
    analytical_derivative = np.full_like(numerical_derivative, fill_value=eval(coefs_prime, delays_prime, s))

    print(numerical_derivative)
    print(analytical_derivative)


    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(eps_vector, np.abs(numerical_derivative), "x-")
    plt.plot(eps_vector, np.abs(analytical_derivative), "r-")
    plt.show()

    fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)

    ax1.plot(eps_vector, np.abs(numerical_derivative.real - analytical_derivative.real))
    ax2.plot(eps_vector, np.abs(numerical_derivative.imag - analytical_derivative.imag))
    plt.show()
    
    
    # plt.figure()
    # plt.semilogy(eps_vector, np.abs(numerical_derivative), "x-")
    # plt.semilogy(eps_vector, np.abs(analytical_derivative), "r-")
    # plt.show()






