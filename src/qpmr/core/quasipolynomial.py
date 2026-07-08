"""Quasi-polynomial core operations
================================

Evaluation, compression, normalization, and representation transforms for
quasi-polynomials in coefficient-delay form."""
import logging
from typing import Any
import math
import warnings
import numpy as np
import numpy.typing as npt
logger = logging.getLogger(__name__)

def _eval_scalar(coefs: npt.NDArray, delays: npt.NDArray, s: int | float | complex):
    """Evaluate a quasi-polynomial at a single complex point."""
    powers = np.arange(0, coefs.shape[1], 1, dtype=int)
    return np.inner(np.sum(coefs * np.power(s, powers), axis=1), np.exp(-delays * s))

def _eval_array(coefs: npt.NDArray, delays: npt.NDArray, s: npt.NDArray):
    """Evaluate a quasi-polynomial on an array of complex points."""
    coefs = coefs.T
    delays = delays
    powers = np.arange(0, coefs.shape[0], 1, dtype=int)
    dels = np.exp(-s[..., np.newaxis] * delays[np.newaxis, ...])
    aa = dels[..., np.newaxis] * coefs.T[np.newaxis, ...]
    r = np.multiply(np.power(s[..., np.newaxis], powers[np.newaxis, ...]), np.sum(aa, axis=-2))
    return np.sum(r, axis=-1)

def _eval_array_opt(coefs: npt.NDArray, delays: npt.NDArray, s: npt.NDArray):
    """Evaluate a quasi-polynomial on an array (optimized power reuse)."""
    _memory = np.ones_like(s)
    dels = np.exp(-s[..., np.newaxis] * delays[np.newaxis, ...])
    result = np.sum(dels * coefs[:, 0], -1)
    for d in range(1, coefs.shape[1]):
        _memory *= s
        result += np.multiply(_memory, np.sum(dels * coefs[:, d], -1))
    return result

def eval(coefs: npt.NDArray, delays: npt.NDArray, s: Any):
    """Evaluate a quasi-polynomial at ``s``.

Parameters
----------
coefs : ndarray
    Matrix of polynomial coefficients. Each row represents the coefficients
    corresponding to a specific delay.
delays : ndarray
    Vector of delays associated with each row in ``coefs``.
s : number or ndarray
    Complex point or array of points.

Returns
-------
value : number or ndarray
    Evaluated quasi-polynomial.

Raises
------
ValueError
    If ``s`` has an unsupported type."""
    if isinstance(s, (int, float, complex)):
        return _eval_scalar(coefs, delays, s)
    elif isinstance(s, np.ndarray):
        return _eval_array(coefs, delays, s)
    else:
        raise ValueError(f"Unsupported type of s '{type(s)}'")

def compress(coefs: npt.NDArray, delays: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """Compress a quasi-polynomial to minimal coefficient-delay form.

Merges duplicate delays, removes zero rows and trailing zero columns, and
sorts delays in ascending order.

Parameters
----------
coefs : ndarray
    Matrix of polynomial coefficients. Each row represents the coefficients
    corresponding to a specific delay.
delays : ndarray
    Vector of delays associated with each row in ``coefs``.

Returns
-------
ccoefs : ndarray
    Compressed coefficient matrix.
ddelays : ndarray
    Compressed delay vector."""
    logger.debug(f'Original quasipolynomial:\n{coefs}\n{delays}')
    if coefs.size == 0 and delays.size == 0:
        return (np.array([[]], dtype=coefs.dtype), np.array([], dtype=delays.dtype))
    delays_compressed = np.unique(delays)
    n, m = (delays_compressed.shape[0], coefs.shape[1])
    coefs_compressed = np.zeros(shape=(n, m), dtype=coefs.dtype)
    for i in range(n):
        mask = delays == delays_compressed[i]
        coefs_compressed[i, :] = np.sum(coefs[mask, :], axis=0, keepdims=False)
    col_mask = ~(coefs_compressed == 0).all(axis=0)
    ix = np.argmax(col_mask[::-1])
    col_mask = np.full_like(col_mask, fill_value=True, dtype=bool)
    if ix > 0:
        col_mask[-ix:] = False
    row_mask = ~(coefs_compressed == 0).all(axis=1)
    coefs_compressed = coefs_compressed[np.ix_(row_mask, col_mask)]
    delays_compressed = delays_compressed[row_mask]
    logger.debug(f'Compressed quasipolynomial\n{coefs_compressed}\n{delays_compressed}')
    if not coefs_compressed.size:
        return (np.array([[]], dtype=coefs.dtype), np.array([], dtype=delays.dtype))
    return (coefs_compressed, delays_compressed)

def normalize(coefs: npt.NDArray, delays: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """ Creates normalized quasi-polynomial representation

    Normalized quasi-polynomial is the quasi-polynomial with the same spectrum
    as the original one, but it's representation is compressed (i.e. unique
    `delays` sorted in ascending order, no ending zero-column in `coefs`). First
    delay is 0.0 and the leading coefficient is 1.0 (leading coefficient is the 
    coefficient with highest power of :math:`s` and smallest delay).

    Parameters
    ----------
    coefs : ndarray
        Matrix of polynomial coefficients. Each row represents the coefficients
        corresponding to a specific delay.

    delays : ndarray
        Vector of delays associated with each row in `coefs`.

    Returns
    -------
    ccoefs : ndarray
        Matrix of normalized polynomial coefficients. Each row represents the
        coefficients corresponding to a specific delay.

    ddelays : ndarray
        Vector of delays associated with each row in `coefs`, non-negative,
        sorted in ascending order, if non-empty first delay is 0

    Examples
    --------
    >>> import numpy as np
    >>> import qpmr.quasipoly
    >>> coefs = np.array([[0., 1, 2, 0], [1, 1, 2, 0], [0, 0, 2, 0]])
    >>> delays = np.array([-1., 1, 1])
    >>> ncoefs, ndelays = qpmr.quasipoly.normalize(coefs, delays)
    >>> ncoefs
    array([[0. , 0.5, 1. ],
           [0.5, 0.5, 2. ]])
    >>> ndelays
    array([0., 2.])

    """
    ccoefs, cdelays = compress(coefs, delays)
    if cdelays.size >= 0:
        cdelays -= np.min(cdelays)
        nonzero_ix = np.nonzero(ccoefs[:, -1])[0]
        ccoefs /= float(ccoefs[nonzero_ix[0], -1])
    return (ccoefs, cdelays)

def normalize_exponent(coefs: npt.NDArray, delays: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray, float]:
    """Normalize delays by scaling time (exponent normalization).

Scales coefficients and delays so the largest delay is 1, preserving the
spectrum up to a time-scale factor.

Parameters
----------
coefs : ndarray
    Matrix of polynomial coefficients. Each row represents the coefficients
    corresponding to a specific delay.
delays : ndarray
    Vector of delays associated with each row in ``coefs``.

Returns
-------
ncoefs : ndarray
    Scaled coefficient matrix.
ndelays : ndarray
    Delays shifted to start at 0 and divided by ``tau_max``.
tau_max : float
    Largest delay before scaling (0 if empty).

Raises
------
NotImplementedError
    If all delays are zero after shifting."""
    if delays.size == 0:
        return (np.copy(coefs), np.copy(delays), 0)
    ndelays = np.copy(delays) - np.min(delays)
    tau_max = np.max(ndelays)
    if tau_max == 0.0:
        raise NotImplementedError
    _, d = coefs.shape
    ncoefs = coefs * np.power(1.0 / tau_max, np.arange(0, d, 1))
    ndelays /= tau_max
    return (ncoefs, ndelays, tau_max)

def factorize_power(coefs: npt.NDArray, delays: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray, int]:
    """Factor out powers of `s` from quasi-polynomial

Where original quasi-polynomial is of a form
.. math::

    h(s) = s^n g(s)

such that :math:`g(s)` is quasi-polynomial such that at least one trailing
coefficient of polynomials associated with :math:`g(s)` is non-zero.


Parameters
----------
coefs : ndarray
    Matrix of polynomial coefficients. Each row represents the coefficients
    corresponding to a specific delay.

delays : ndarray
    Vector of delays associated with each row in `coefs`.

Returns
-------
ccoefs : ndarray
    Matrix of normalized polynomial coefficients. Each row represents the
    coefficients corresponding to a specific delay.

ddelays : ndarray
    Vector of delays associated with each row in `coefs`, non-negative,
    sorted in ascending order, if non-empty first delay is 0

spower : int
    Exponent :math:`n` factored out from ``s``.

Examples
--------
>>> import numpy as np
>>> import qpmr.quasipoly as qp
>>> coefs = np.array([[0., 1.], [1., 0.]])
>>> delays = np.array([0., 1.])
>>> g_coefs, g_delays, n = qp.factorize_power(coefs, delays)
>>> n
1"""
    if delays.size == 0:
        return (np.copy(coefs), np.copy(delays), 0)
    n, d = coefs.shape
    mask = np.any(coefs != 0.0, axis=0)
    if not np.any(mask):
        return (np.empty(shape=(0, 0), dtype=coefs.dtype), np.array([], dtype=delays.dtype), 0)
    ix = np.argmax(mask)
    ccoefs = np.copy(coefs[:, ix:])
    ddelays = np.copy(delays)
    return (ccoefs, ddelays, ix)

def shift(coefs: npt.NDArray, delays: npt.NDArray, origin: float | complex) -> tuple[npt.NDArray, npt.NDArray]:
    """Shifts quasi-polynomial to new origin

Input quasi-polynomial :math:`h(s)` is shifted to new origin
:math:`a` resulting in quasi-polynomial

.. math::

    g(s) = h(s-a)

Parameters
----------
coefs : ndarray
    Matrix of polynomial coefficients. Each row represents the coefficients
    corresponding to a specific delay.

delays : ndarray
    Vector of delays associated with each row in `coefs`.

origin : float or complex
    New origin of complex plane

Returns
-------
ccoefs : ndarray
    Matrix of shifted polynomial coefficients. Each row represents the
    coefficients corresponding to a specific delay.

ddelays : ndarray
    Vector of delays associated with each row in `coefs`, it is identical
    to input

Examples
--------
>>> import numpy as np
>>> import qpmr.quasipoly
>>> coefs = np.array([[0., 1], [0, 1]])
>>> delays = np.array([0., 1])
>>> ccoefs, ddelays = qpmr.quasipoly.shift(coefs, delays, 1)
>>> ccoefs
array([[-1.        ,  1.        ],
       [-2.71828183,  2.71828183]])

Try to shift back to the original origin and check the results

>>> ccoefs, ddelays = qpmr.quasipoly.shift(ccoefs, ddelays, -1)
>>> ccoefs
array([[0., 1.],
       [0., 1.]])"""
    if isinstance(origin, complex):
        dtype = np.complex128
    else:
        dtype = coefs.dtype
    n, d = coefs.shape
    shift_matrix = np.zeros(shape=(d, d), dtype=dtype)
    for i in range(d):
        for j in range(i + 1):
            shift_matrix[i, j] = math.comb(i, j) * (-origin) ** (i - j)
    ccoefs = coefs @ shift_matrix * np.exp(origin * delays)[:, None]
    ddelays = np.copy(delays)
    return (ccoefs, ddelays)

def poly_degree(poly: npt.NDArray, order='reversed') -> int:
    """Return the degree of a 1D polynomial coefficient vector.

Parameters
----------
poly : ndarray
    Coefficient vector, possibly with trailing zeros.
order : {'reversed', 'forward'}, optional
    If ``'reversed'``, highest degree is the last non-zero entry when read
    from the end. Default is ``'reversed'``.

Returns
-------
degree : int
    Polynomial degree."""
    degree = len(poly) - 1
    if order == 'reversed':
        poly_ = poly[::-1]
    else:
        poly_ = poly
    for a in poly_:
        if a != 0.0:
            break
        degree -= 1
    logger.debug(f'poly={poly!r} -> degree: {degree}')
    return degree

def is_neutral(coefs: npt.NDArray, delays: npt.NDArray, **kwargs):
    """Test whether a quasi-polynomial is of neutral type.

Parameters
----------
coefs : ndarray
    Matrix of polynomial coefficients. Each row represents the coefficients
    corresponding to a specific delay.
delays : ndarray
    Vector of delays associated with each row in ``coefs``.
compress : bool, optional
    If ``True`` (default), compress before the test.

Returns
-------
neutral : bool
    ``True`` if neutral, ``False`` if retarded or empty."""
    if kwargs.get('compress', True):
        coefs, delays = compress(coefs, delays)
    if coefs.size == 0 and delays.size == 0:
        return False
    if len(delays) > 1 and coefs[0, -1] != 0:
        return True
    else:
        return False

def extract_delay_diff_eq(coefs: npt.NDArray, delays: npt.NDArray, **kwargs) -> tuple[npt.NDArray, npt.NDArray]:
    """Extract the delay-difference equation from a quasi-polynomial.

Uses the highest-power coefficients (last column of ``coefs``).

Parameters
----------
coefs : ndarray
    Matrix of polynomial coefficients. Each row represents the coefficients
    corresponding to a specific delay.
delays : ndarray
    Vector of delays associated with each row in ``coefs``.
compress : bool, optional
    If ``True``, compress first. Default is ``False``.
normalize : bool, optional
    If ``True`` (default), divide by the leading coefficient and shift
    delays so the first is zero.

Returns
-------
diff_coefs : ndarray
    1D coefficients of the delay-difference equation.
diff_delays : ndarray
    Associated delays."""
    if kwargs.get('compress', False):
        coefs, delays = compress(coefs, delays)
    diff_coefs = coefs[:, -1]
    mask = diff_coefs != 0.0
    diff_coefs = diff_coefs[mask]
    diff_delays = delays[mask]
    if len(diff_coefs) == 0:
        warnings.warn(f'Quasipolynomial: P={coefs.tolist()} delays={delays.tolist()} is not compressed or is empty after compresion, returning empty arrays')
        return (np.empty_like(diff_coefs), np.empty_like(delays))
    if kwargs.get('normalize', True):
        diff_coefs /= diff_coefs[0]
    diff_delays -= diff_delays[0]
    return (diff_coefs, diff_delays)

def create_normalized_delay_difference_eq(coefs: npt.NDArray, delays: npt.NDArray, **kwargs) -> tuple[npt.NDArray, npt.NDArray]:
    """Build the normalized delay-difference characteristic equation.

For neutral quasi-polynomials, returns absolute values of sub-leading
coefficients and their delays (leading term normalized to 1 and omitted).

Parameters
----------
coefs : ndarray
    Matrix of polynomial coefficients. Each row represents the coefficients
    corresponding to a specific delay.
delays : ndarray
    Vector of delays associated with each row in ``coefs``.
compress : bool, optional
    If ``True`` (default), compress before extraction.

Returns
-------
diff_coefs : ndarray or None
    Normalized delay-difference coefficients, or ``None`` if the
    quasi-polynomial is not neutral.
diff_delays : ndarray or None
    Delays for ``diff_coefs``, or ``None``."""
    if kwargs.get('compress', True):
        coefs, delays = compress(coefs, delays)
    if is_neutral(coefs, delays, compress=False):
        a0 = coefs[0, -1]
        return (np.abs(coefs[1:, -1] / a0), delays[1:])
    else:
        return None