r"""
Numerical root refinement
=========================

Newton, secant, and Müller methods used to polish spectrum-mapping guesses.
"""

from .newton_method import newton, numerical_newton
from .secant_method import secant
from .mueller_method import mueller

__all__ = ["newton", "numerical_newton", "secant", "mueller"]
