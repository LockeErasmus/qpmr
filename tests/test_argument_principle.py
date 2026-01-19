"""

"""

import pytest

import qpmr

from qpmr.argument_principle import circle_contour, rectangular_contour, _argument_principle, _argument_principle_robust
from qpmr.quasipoly.core import eval
import qpmr.quasipoly.examples as examples



@pytest.mark.parametrize(
    argnames="qp, contour, expected, params",
    argvalues=[
        (
            examples.vyhlidal2014qpmr,
            rectangular_contour(-13.040391824936004, 0.8813024303938415, -0.3141592653589793, 628.6326899833176),
            100,
            {
                "example":{"args": (1,), "kwargs": {}},
            },
        ),
        (
            examples.mazanti2021multiplicity,
            circle_contour(-6.021, 0.1),
            6,
            {},
        ),
        (
            examples.mazanti2021multiplicity,
            rectangular_contour(-15.08, 1.08, -0.08, 50.08),
            8,
            {}
        ),
        (
            examples.mazanti2021multiplicity,
            rectangular_contour(-15., 1., 0., 50),
            7,
            {}
        ),
        (
            examples.self_inverse_polynomial(0, 0.01, 8),
            circle_contour(0., 0.01001),
            8,
            {},
        )
    ],
    
    ids=[
        "vyhlidal2014qpmr_01_rectangle01",
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
        args = params.get("example", {}).get("args", ())
        kwargs = params.get("example", {}).get("kwargs", {})
        coefs, delays = qp(*args, **kwargs)
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
    n_robust = _argument_principle_robust(f, gamma, a, b, n_points_0=1000)

    print(f"Argument principle {n=}, {n_robust=}, {expected=}")

    


    
    
    
    