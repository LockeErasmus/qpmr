"""
TODO:
    1. is_empty QP
"""

import logging

import numpy as np
import numpy.typing as npt
import scipy.signal


logger = logging.getLogger(__name__)

def compress_qp(coefs: npt.NDArray, delays: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
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

class QuasiPolynomial:

    def __init__(self, coefs: npt.NDArray, delays: npt.NDArray) -> None:
        
        # TODO ASSERT 2D
        # TODO match n, ...
        # coefs has to be 2D, if empty, shape has to be (0,0)

        self.coefs = coefs
        self.delays = delays

    
    def __call__(self, s: complex):
        ... # TODO
        # consider also, value, value1D, value2D, valueND

    @property
    def is_empty(self) -> bool:
        """ Checks if qp empty, equivalent to p(s) = 0 """
        logger.debug(f"size={self.coefs.size}  |  {not bool(self.coefs.size)}")
        if self.coefs.size:
            return False
        else:
            return True
    
    @property
    def degree(self) -> int:
        """ maximal degree polynomials """
        return self.m - 1
    
    @property
    def m(self):
        """ number of powers = degree + 1 for non is_empty """
        if self.is_empty:
            return 0
        else:
            return self.coefs.shape[1]

    @property
    def n(self):
        """ number of delays """
        if self.is_empty:
            return 0
        else:
            return self.coefs.shape[0]
    
    @property
    def is_constant(self) -> bool:
        raise NotImplementedError("")

    @property
    def is_polynomial(self) -> bool:
        raise NotImplementedError("")
    
    @property
    def is_retarded(self) -> bool:
        """ checks if qp is of retarded type """
        raise NotImplementedError("")

    @property
    def is_neutral(self) -> bool:
        raise NotImplementedError("")

    @property
    def is_advanced(self) -> bool:
        raise NotImplementedError("")
    
    @property
    def derivative(self) -> 'QuasiPolynomial':
        # TODO solve empty and constant cases
        if self.is_empty:
            # derivative of empty quasi-polynomial is empty quasipolynomial
            return QuasiPolynomial(
                coefs=np.array([[]], dtype=self.coefs.dtype),
                delays=np.array([], dtype=self.delays.dtype),
            )
        
        order_vector = np.arange(1, self.m, 1)
        coefs = - self.coefs * self.delays[:, np.newaxis]        
        coefs[:, :-1] = coefs[:, :-1] + self.coefs[:, 1:] * order_vector    
        return QuasiPolynomial(coefs=coefs, delays=self.delays)

    def minimal_form(self) -> 'QuasiPolynomial':
        """ Converts QuasiPolynomial to minimal sorted form with no duplicate delays
        """
        logger.debug(f"Minimal form of QP\n{self.coefs}\n{self.delays}")
        new_delays = np.unique(self.delays) # sorted 1D array of unique delays
        new_n = new_delays.shape[0] # new number of delays
        new_coefs = np.zeros(shape=(new_n, self.m), dtype=self.coefs.dtype)
        
        for i in range(new_n):
            mask = self.delays == new_delays[i]
            new_coefs[i, :] = np.sum(self.coefs[mask, :], axis=0, keepdims=False)

        # remove all zero column or rows
        mask = new_coefs == 0
        col_mask = ~mask.all(0)
        row_mask = ~mask.all(1) 
        new_coefs = new_coefs[np.ix_(row_mask, col_mask)]
        new_delays = new_delays[row_mask]

        qp = QuasiPolynomial(new_coefs, new_delays)
        logger.debug(f"Minimal form of QP\n{qp.coefs}\n{qp.delays}")

        return qp
    
    @property
    def poly_degrees(self) -> npt.NDArray:
        # TODO is_empty
        return np.apply_along_axis(poly_degree, 1, self.coefs)
    
    def eval(self, x):
        """ Evaluates quasipolynomial at x """
        if isinstance(x, (int, float, complex)):
            coefs = self.coefs # transpose rows - powers of s, cols - delays
            delays = self.delays
            powers = np.arange(0, coefs.shape[1], 1, dtype=int)
            return np.inner(np.sum(coefs * np.power(x,  powers), axis=1), np.exp(-delays*x))
        elif isinstance(x, np.ndarray):
            coefs = self.coefs.T # transpose rows - powers of s, cols - delays
            delays = self.delays
            powers = np.arange(0, coefs.shape[0], 1, dtype=int)
            dels = np.exp(- x[..., np.newaxis] * delays[np.newaxis, ...])
            aa = dels[..., np.newaxis] * coefs.T[np.newaxis, ...] # (..., n_delays, order)
            r = np.multiply(
                np.power(x[..., np.newaxis], powers[np.newaxis, ...]), # (..., order)
                np.sum(aa, axis=-2), # sum by n_delays axis -> (..., order)
            )
            return np.sum(r, axis=-1)
        else:
            raise ValueError(f"Unsupported type of x '{type(x)}'")
    
    def __neg__(self):
        return QuasiPolynomial(-self.coefs, self.delays)
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            const_qp = QuasiPolynomial(
                np.array([[other]], dtype=self.coefs.dtype),
                np.array([0.], dtype=self.delays.dtype),
            )
            return self.__add__(const_qp)
        elif isinstance(other, QuasiPolynomial):
            if self.m >= other.m:
                a = np.zeros(shape=(other.n, self.m), dtype=other.coefs.dtype)
                a[:other.coefs.shape[0], :other.coefs.shape[1]] = other.coefs
                coefs = np.r_[self.coefs, a]
                delays = np.r_[self.delays, other.delays]
                return QuasiPolynomial(coefs, delays)
            else:
                return other.__add__(self)
        else:
            raise Exception("Not possible to add TODO")
    
    def __sub__(self, other):
        return self.__add__(self, -other)
    
    def __pow__(self, other):
        raise NotImplementedError("")

    def __mul__(self, other):
        """ """
        if isinstance(other, (int, float)):
            return QuasiPolynomial(other * self.coefs, self.delays)
        elif isinstance(other, QuasiPolynomial):
            # TODO emptyness
            coef_list = []
            delay_list = []
            for j in range(other.m):
                coef_list.append(
                    scipy.signal.convolve2d(self.coefs, other.coefs[j:j+1,:], mode="fill", boundary="fill")
                )
                delay_list.append(self.delays + other.delays[j])
            
            new_coefs = np.concatenate(coef_list, axis=0)
            new_delays = np.concatenate(delay_list)
            return QuasiPolynomial(new_coefs, new_delays)
        else:
            raise NotImplementedError(f"Not possible to raise quasi-polynomial to {type(other)}")
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __rsub__(self, other):
        return self.__sub__(other)

    def __rmul__(self, other):
        return self.__mul__(other)
    
if __name__ == "__main__":

    delays = np.array([0.0, 1.0, 0.0, 1.0, 2.5])
    coefs = np.array([[0, 1],
                      [1, 0],
                      [0, 2],
                      [3, 0],
                      [0, 0]])

    qp = QuasiPolynomial(coefs, delays)

    print(qp.coefs)
    print(qp.delays)

    qp = qp.minimal_form()

    print(qp.coefs)
    print(qp.delays)
