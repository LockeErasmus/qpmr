"""

"""

import pytest

import qpmr

from qpmr.argument_principle import circle_contour, rectangular_contour, _argument_principle
from qpmr.quasipoly.core import eval
from qpmr.quasipoly.examples import mazanti2021multiplicity, self_inverse_polynomial



@pytest.mark.parametrize(
    argnames="qp, contour, expected, params",
    argvalues=[
        (
            mazanti2021multiplicity,
            circle_contour(-6.021, 0.1),
            6,
            {},
        ),
        (
            mazanti2021multiplicity,
            rectangular_contour(-15.08, 1.08, -0.08, 50.08),
            8,
            {}
        ),
        (
            mazanti2021multiplicity,
            rectangular_contour(-15., 1., 0., 50),
            7,
            {}
        ),
        (
            self_inverse_polynomial(0, 0.01, 8),
            circle_contour(0., 0.01001),
            8,
            {
                "use_analytical_derivative": False,
            },
        )
    ],
    
    ids=[
        "mazanti2021multiplicity_circ01",
        "mazanti2021multiplicity_rect01",
        "mazanti2021multiplicity_rect02",
        "Disk(0, 0.01, 8)",
    ],
)
def test_argument_principle(qp, contour, expected: int, params):

    if isinstance(qp, qpmr.QuasiPolynomial):
        coefs, delays = qp.coefs, qp.delays
    elif callable(qp):
        coefs, delays = qp()
    elif isinstance(qp, tuple):
        coefs, delays = qp

    f = lambda s: eval(coefs, delays, s)
    if True: # numerical derivative
        def f_prime(s, eps=1e-8):
            dvals = (f(s - eps) - f(s + eps) + 1j*f(s +1j*eps)
                     - 1j*f(s -1j*eps)) / 4. / eps
            return dvals        
    else: # analytical derivative
        raise NotImplementedError(".")
        df = None # TODO

    gamma, gamma_prime, (a,b) = contour
    
    n = _argument_principle(f, f_prime, gamma, gamma_prime, a, b)

    print(f"Argument principle {n=}, {expected=}")

    


    
    
    
    