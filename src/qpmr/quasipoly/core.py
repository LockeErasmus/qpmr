"""
TODO:
    1. empty QP
"""

import logging

import numpy as np
import numpy.typing as npt


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
        # TODO checks 2D, match n, ...

        self.coefs = coefs
        self.delays = delays
    
    def __call__(self, s: complex):
        ... # TODO
        # consider also, value, value1D, value2D, valueND
    
    @property
    def degree(self) -> int:
        """ maximal degree polynomials """
        if self.empty:
            return 0
        else:
            return self.coefs.shape[1] - 1
    
    @property
    def m(self):
        """ number of powers = degree + 1 for non empty """
        return self.coefs.shape[1]

    @property
    def n(self):
        """ number of delays """
        return self.coefs.shape[0]
    
    @property
    def is_retarded(self) -> bool:
        pass

    @property
    def is_neutral(self) -> bool:
        pass

    @property
    def is_advanced(self) -> bool:
        return np.any(self.delays < 0.0)

    @property
    def empty(self) -> bool:
        """ equivalent to p(s) = 0
        """
        return False # TODO

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
        # TODO empty
        return np.apply_along_axis(poly_degree, 1, self.coefs)
    
    def __neg__(self):
        return QuasiPolynomial(-self.coefs, self.delays)
    
    def __add__(self, other):
        if isinstance(other, float):
            pass
        elif isinstance(other, QuasiPolynomial):
            degree = max(self.degree, other.degree)
            coefs = 0 
        else:
            raise Exception("Not possible to add TODO")
    
    def __sub__(self, other):
        raise NotImplementedError("")
    
    def __pow__(self, other):
        raise NotImplementedError("")

    def __mul__(self, other):
        if isinstance(other, float):
            return QuasiPolynomial(other * self.coefs, self.delays)
        else:
            raise NotImplementedError("mul")
    
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
