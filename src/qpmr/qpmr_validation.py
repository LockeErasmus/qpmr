r"""
Input validation (re-exports)
=============================

Re-exports validation helpers from :mod:`qpmr.core.validation` for use by the
top-level QPmR API.
"""

from .core.validation import validate_qp, validate_region

__all__ = ["validate_qp", "validate_region"]
