"""
"""
import logging

import matplotlib.pyplot as plt
import numpy as np

from qpmr.quasipoly import QuasiPolynomial

_ = logging.getLogger("matplotlib").setLevel(logging.ERROR)
_ = logging.getLogger("PIL").setLevel(logging.ERROR)
logger = logging.getLogger("qpmr")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    delays = np.array([1.0, 1.0, 1.0])
    coefs = np.array([[0, -1, 1.5, 0, 0, -10],
                      [0, 0.3, -0.5, 0, 0, -10],
                      [0, 0.7, -1, 0, 0, 20]])
    qp = QuasiPolynomial(coefs, delays)
    qp_minimal = qp.minimal_form()

    print(f"emtpy = {qp_minimal.is_empty}")


    qp2 = qp_minimal * 6
    qp2.minimal_form()

    delays = np.array([0.0, 1.0])
    coefs = np.array([[0, 1],[1, 0]])
    qp3 = QuasiPolynomial(coefs, delays)

    print(qp2.coefs.shape)
    qp4 = qp3 + qp3

    qp4.minimal_form()


    empty_qp = QuasiPolynomial(
        np.empty(shape=(0,0), dtype=np.float64),
        np.empty(shape=(0,), dtype=np.float64),
    )
    qp = QuasiPolynomial(
        np.array([[0, 1],[1, 0]], dtype=np.float64),
        np.array([0.0, 1.0], dtype=np.float64),
    )

    print(empty_qp.is_empty)
    print(empty_qp.coefs.shape)
    qp_result = qp + empty_qp


