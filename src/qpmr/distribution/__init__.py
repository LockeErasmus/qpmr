r"""
Distribution analysis
=====================

Spectrum distribution diagrams, zero-chain asymptotics, and spectral bounds.
"""

from .spectrum_distribution_diagram import distribution_diagram
from .zero_chains import chain_asymptotes
from .spectral_abscissa import safe_upper_bound_diff, bounds_neutral_strip

__all__ = [
    "distribution_diagram",
    "chain_asymptotes",
    "safe_upper_bound_diff",
    "bounds_neutral_strip",
]
