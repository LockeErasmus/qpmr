r"""
Quasi-polynomial package
========================

Coefficient-delay representation, arithmetic, and object wrappers.
"""

from .obj import QuasiPolynomial, TransferFunction
from .core import compress, normalize, eval, shift, normalize_exponent, factorize_power
from .arithmetic import add, multiply
from .operation import derivative, antiderivative

__all__ = [
    "QuasiPolynomial",
    "TransferFunction",
    "compress",
    "normalize",
    "eval",
    "shift",
    "normalize_exponent",
    "factorize_power",
    "add",
    "multiply",
    "derivative",
    "antiderivative",
]
