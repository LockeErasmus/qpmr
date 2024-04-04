"""
TODO:
    1. is_empty QP
"""

import logging

import numpy as np
import numpy.typing as npt
import scipy.signal


logger = logging.getLogger(__name__)

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

    def __init__(self, coefs: npt.NDArray, delays) -> None:
        
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
