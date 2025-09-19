"""
TODO:
    1. is_empty QP
"""

import logging
from typing import Any

import math
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

def _eval_scalar(coefs: npt.NDArray, delays: npt.NDArray, s: int|float|complex):
    """ Evaluates quasipolynomial on complex value

    Args:
        coefs (array): matrix definition of polynomial coefficients (each row
            represents polynomial coefficients corresponding to delay)
        delays (array): vector definition of associated delays (each delay
            corresponds to row in `coefs`)
        s (number): complex value to evaluate quasipolynomial

    Returns:
        number: evaluated quasipolynomial at `s`
    """
    powers = np.arange(0, coefs.shape[1], 1, dtype=int)
    return np.inner(np.sum(coefs * np.power(s,  powers), axis=1), np.exp(-delays*s))

def _eval_array(coefs: npt.NDArray, delays: npt.NDArray, s: npt.NDArray):
    """ Evaluates quasipolynomial on nD array of complex values

    Args:
        coefs (array): matrix definition of polynomial coefficients (each row
            represents polynomial coefficients corresponding to delay)
        delays (array): vector definition of associated delays (each delay
            corresponds to row in `coefs`)
        s (array): nD array of complex values to evaluate quasipolynomial on

    Returns:
        array: evaluated quasipolynomial at all elements of `s`
    """
    coefs = coefs.T # transpose rows - powers of s, cols - delays
    delays = delays
    powers = np.arange(0, coefs.shape[0], 1, dtype=int)
    dels = np.exp(- s[..., np.newaxis] * delays[np.newaxis, ...])
    aa = dels[..., np.newaxis] * coefs.T[np.newaxis, ...] # (..., n_delays, order)
    r = np.multiply(
        np.power(s[..., np.newaxis], powers[np.newaxis, ...]), # (..., order)
        np.sum(aa, axis=-2), # sum by n_delays axis -> (..., order)
    )
    return np.sum(r, axis=-1)

def _eval_array_opt(coefs: npt.NDArray, delays: npt.NDArray, s: npt.NDArray):
    """ Evaluates quasipolynomial on nD array of complex values OPTIMIZED

    Leverages the fact that:
        s^(k+1) = s * s^k,
    i.e. it reuses powers of s in calculation.

    Args:
        coefs (array): matrix definition of polynomial coefficients (each row
            represents polynomial coefficients corresponding to delay)
        delays (array): vector definition of associated delays (each delay
            corresponds to row in `coefs`)
        s (array): nD array of complex values to evaluate quasipolynomial on

    Returns:
        array: evaluated quasipolynomial at all elements of `s`
    """
    _memory = np.ones_like(s)
    # solve degree 0
    dels = np.exp(- s[..., np.newaxis] * delays[np.newaxis, ...])
    result = np.sum(dels * coefs[:, 0], -1)
    for d in range(1, coefs.shape[1]): # solve other degrees
        _memory *= s # elementwise multiplication
        result += np.multiply(_memory, np.sum(dels * coefs[:, d], -1))
    return result

def eval(coefs: npt.NDArray, delays: npt.NDArray, s: Any):
    """ Evaluates quasipolynomial on s

    Args:
        coefs (array): matrix definition of polynomial coefficients (each row
            represents polynomial coefficients corresponding to delay)
        delays (array): vector definition of associated delays (each delay
            corresponds to row in `coefs`)
        s (array or number): complex value (s) to evaluate quasipolynomial on

    Returns:
        array: evaluated quasipolynomial at all elements of `s`
    """
    if isinstance(s, (int, float, complex)):
        return _eval_scalar(coefs, delays, s)
    elif isinstance(s, np.ndarray):
        return _eval_array(coefs, delays, s)
    else:
        raise ValueError(f"Unsupported type of s '{type(s)}'")

def compress(coefs: npt.NDArray, delays: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """ Compresses quasipolynomial representation into a form where no
    duplicates in delays are present and last column vector of `coefs` is
    non-zero. Compressed quasipolynomial has also ordered delays in ascending
    order.

    Args:
        coefs (array): matrix definition of polynomial coefficients (each row
            represents polynomial coefficients corresponding to delay)
        delays (array): vector definition of associated delays (each delay
            corresponds to row in `coefs`)
    
    Returns:
        tuple containing:

            - coefs (array): matrix definition of polynomial coefficients (each row
                represents polynomial coefficients corresponding to delay)
            - delays (array): vector definition of associated delays (each delay
                corresponds to row in `coefs`)

    """
    logger.debug(f"Original quasipolynomial:\n{coefs}\n{delays}")
    if coefs.size == 0 and delays.size == 0: # trivial case with empty definition, i.e. qp(s) = 0
        # note that new arrays are constructed, because if coefs are of shape
        # for instance (N,0) I want always the shape to be (1,0)
        return np.array([[]], dtype=coefs.dtype), np.array([], dtype=delays.dtype)
    
    delays_compressed = np.unique(delays) # sorted 1D array of unique delays
    n, m = delays_compressed.shape[0], coefs.shape[1]
    coefs_compressed = np.zeros(shape=(n, m), dtype=coefs.dtype)
    
    for i in range(n):
        mask = (delays == delays_compressed[i])
        coefs_compressed[i, :] = np.sum(coefs[mask, :], axis=0, keepdims=False)
    
    # at this point, representation is unique in delays, we need to make sure
    # `coefs_compressed` does not have: 1) row full of zeros and 2) last column
    #  full of zeros
    col_mask = ~(coefs_compressed == 0).all(axis=0) # True if column has at least one non-zero
    ix = np.argmax(col_mask[::-1]) # first occurence of True indexed from end
    col_mask = np.full_like(col_mask, fill_value=True, dtype=bool)
    if ix > 0: # at least one column from back should be deleted
        col_mask[-ix:] = False
    row_mask = ~(coefs_compressed == 0).all(axis=1) # True if row has atleast one non-zero coefficient

    coefs_compressed = coefs_compressed[np.ix_(row_mask, col_mask)]
    delays_compressed = delays_compressed[row_mask]
    logger.debug((f"Compressed quasipolynomial\n{coefs_compressed}"
                  f"\n{delays_compressed}"))
    if not coefs_compressed.size: # resulting qp is empty
        return np.array([[]], dtype=coefs.dtype), np.array([], dtype=delays.dtype)

    return coefs_compressed, delays_compressed

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
    >>> coefs
    array([[0. , 0.5, 1. ],
           [0.5, 0.5, 2. ]])
    >>> delays
    array([0., 2.])

    """
    ccoefs, cdelays = compress(coefs, delays)

    # cdelays.size == 0 --> trivial case representing h(s) = 0
    if cdelays.size >= 0:
        cdelays -= np.min(cdelays)
        nonzero_ix = np.nonzero(ccoefs[:,-1])[0] # this should be always non-empty!
        ccoefs /= float(ccoefs[nonzero_ix[0], -1]) # this should be always non-zero
    
    return ccoefs, cdelays

def shift(coefs: npt.NDArray, delays: npt.NDArray, origin: float|complex) -> tuple[npt.NDArray, npt.NDArray]:
    """ Shifts quasi-polynomial to new origin

    Input quasi-polynomial :math:`h(s)` is shifted to new origin
    :math:`a` resulting in quasi-polynomial

    ..math::

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
           [0., 1.]])

    """
    # TODO solve trivial cases like empty... etc
    # TODO solve complex origin
    if isinstance(origin, complex):
        dtype = np.complex128
    else:
        dtype = coefs.dtype

    n, d = coefs.shape
    
    # prepare shift matrix T - TODO separate function?
    shift_matrix = np.zeros(shape=(d, d), dtype=dtype)
    for i in range(d): # rows
        for j in range(i+1): # columns
            shift_matrix[i,j] = math.comb(i, j) * (-origin)**(i-j)
    
    ccoefs = (coefs @ shift_matrix) * np.exp(origin*delays)[:, None]
    ddelays = np.copy(delays)

    return ccoefs, ddelays

def poly_degree(poly: npt.NDArray, order="reversed") -> int:
    """ assumes 1D array as input
    
    [a0, a1, ... am, 0, 0, ... 0] -> degree m

    reverse order

    """
    degree = len(poly) - 1
    if order == "reversed":
        poly_ = poly[::-1]
    else:
        poly_ = poly
    for a in poly_:
        if a != 0.:
            break
        degree -= 1
    logger.debug(f"{poly=} -> degree: {degree}")
    return degree


def is_neutral(coefs: npt.NDArray, delays: npt.NDArray, **kwargs):
    """ Checks if quasipolynomial is neutral
    
    Args:
        coefs (array): matrix definition of polynomial coefficients (each row
            represents polynomial coefficients corresponding to delay)
        delays (array): vector definition of associated delays (each delay
            corresponds to row in `coefs`)
        **kwargs:
            compress (bool): if True compresses the result (converts to minimal
                form), default True
    
    Returns:
        bool - True if neutral, False if retarded
    """

    if kwargs.get("compress", True):
        coefs, delays = compress(coefs, delays)
    
    # assume compressed form here, i.e. delays[0] is 0 (or smallest positive delay)
    # first, solve empty
    if coefs.size == 0 and delays.size == 0:
        return False # this is not even quasipolynomial, just polynomial p(s) = 0
    
    if len(delays) > 1 and coefs[0,-1] != 0:
        return True
    else:
        return False

def create_normalized_delay_difference_eq(coefs: npt.NDArray, delays: npt.NDArray, **kwargs) -> tuple[npt.NDArray, npt.NDArray]:
    """ Creates characteristic equation of the associated delay difference
    equation

    The characteristic equation is normalized (leading term = 1 and is omitted)
        
    returns None if does not exist

    TODO
    """
    if kwargs.get("compress", True):
        coefs, delays = compress(coefs, delays)

    # assume compressed form here, i.e. delays[0] is 0 (or smallest positive delay)
    # first, check neutrality
    if is_neutral(coefs, delays, compress=False):
        a0 = coefs[0,-1] # a0 =/= 0 IFF quasipolynomial is neutral
        return np.abs(coefs[1:,-1]/a0), delays[1:]
    else:
        return None
