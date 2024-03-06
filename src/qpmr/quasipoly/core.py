"""
TODO:
    1. empty QP
"""

import logging

import numpy as np
import numpy.typing as npt


logger = logging.getLogger(__name__)

class QuasiPolynomial:

    def __init__(self, coefs: npt.NDArray, delays) -> None:
        # TODO checks 2D, match n, ...

        self.coefs = coefs
        self.delays = delays
    
    @property
    def degree(self):
        """ maximal degree polynomials """
        return self.coefs.shape[1]

    @property
    def n(self):
        """ number of delays """
        return self.coefs.shape[0]
    
    def minimal_form(self) -> 'QuasiPolynomial':
        """ Converts QuasiPolynomial to minimal form with no duplicate delays
        """
        new_delays = np.unique(self.delays) # sorted 1D array of unique delays
        new_n = new_delays.shape[0] # new number of delays
        new_coefs = np.zeros(shape=(new_n, self.degree), dtype=self.coefs.dtype)
        
        for i in range(new_n):
            mask = self.delays == new_delays[i]
            new_coefs[i, :] = np.sum(self.coefs[mask, :], axis=0, keepdims=False)

        # remove all zero column or rows
        mask = new_coefs == 0
        col_mask = ~mask.all(0)
        row_mask = ~mask.all(1) 
        new_coefs = new_coefs[np.ix_(row_mask, col_mask)]
        new_delays = new_delays[row_mask]

        return QuasiPolynomial(new_coefs, new_delays)
    
    def __neg__(self):
        return QuasiPolynomial(-self.coefs, self.delays)
    
    def __add__(self, other):
        raise NotImplementedError("")
    
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
