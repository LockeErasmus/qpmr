Docstring Style Guide
=====================

The **qpmr** package uses `NumPy-style docstrings
<https://numpydoc.readthedocs.io/en/latest/format.html>`_ for all public
APIs. Sphinx renders them via the Napoleon extension (see
``docs/source/conf.py``).

Format
------

Each docstring should contain, in order:

1. **Summary line** — one line ending without a period when possible.
2. **Extended description** — optional paragraph(s) with algorithm context.
3. **Parameters** — arguments and keyword options.
4. **Returns** — return values (omit for ``None``).
5. **Raises** — exceptions raised by the function (when applicable).
6. **Notes** — math, implementation details, edge cases.
7. **Examples** — runnable ``>>>`` doctest blocks.
8. **References** — literature citations when algorithms are referenced.

Classes and dataclasses should also document public **Attributes** (or use
``Attributes`` for dataclass fields in the class docstring).

Section headers use underlines of dashes::

    Parameters
    ----------
    coefs : ndarray
        ...

Standard parameter descriptions
-------------------------------

Reuse these descriptions verbatim for quasi-polynomial arguments:

``coefs`` : ndarray
    Matrix of polynomial coefficients. Each row represents the coefficients
    corresponding to a specific delay.

``delays`` : ndarray
    Vector of delays associated with each row in ``coefs``.

``region`` : tuple of float
    Rectangular region in the complex plane as
    ``(Re_min, Re_max, Im_min, Im_max)``.

Type hints
----------

Keep type annotations in function signatures. Sphinx is configured with
``autodoc_typehints = 'description'``, so types appear in the built docs from
signatures. Docstrings should describe semantics, not repeat type names.

Mathematics
-----------

Use inline math with ``:math:`` and display math with a proper directive::

    .. math::

        h(s) = \sum_{i=0}^n p_i(s) e^{-s \tau_i}

Note the space after ``..`` in ``.. math::``.

Examples
--------

Prefer doctest-style examples that import the public API::

    >>> import numpy as np
    >>> import qpmr
    >>> coefs = np.array([[0., 1.], [1., 0.]])
    >>> delays = np.array([0., 1.])

Module docstrings
-----------------

Module-level docstrings use an RST title and a short purpose paragraph::

    r"""
    Region heuristic
    ================

    Methods to propose a search region for quasi-polynomial root finding.
    """

Private APIs
------------

Functions and methods whose names start with ``_`` are internal. They may use
a one-line summary only. Sphinx excludes them by default
(``napoleon_include_private_with_doc = False``).
