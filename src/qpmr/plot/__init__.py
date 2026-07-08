r"""
Plotting utilities (requires matplotlib)
==========================================

Visualization of roots, spectrum distribution diagrams, and QPmR diagnostics.
"""

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError("Install matplotlib package via pip to use qpmr.plot")

from .basic import roots, pole_zero, qpmr_contour, argument_principle_circle
from .delay_distribution import spectrum_distribution_diagram, chain_asymptotes, spectrum_envelope
from .complex import experimental
from .solution import qpmr_solution_tree

__all__ = [
    "roots",
    "pole_zero",
    "qpmr_contour",
    "argument_principle_circle",
    "spectrum_distribution_diagram",
    "chain_asymptotes",
    "spectrum_envelope",
    "experimental",
    "qpmr_solution_tree",
]
