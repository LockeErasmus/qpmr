"""

"""

import logging
from typing import Any

import numpy as np
import numpy.typing as npt

from .arithmetic import add, multiply
from .core import compress, eval


logger = logging.getLogger(__name__)

class QuasiPolynomial:

    def __init__(self, coefs: npt.NDArray, delays: npt.NDArray) -> None:
        
        # TODO ASSERT 2D
        # TODO match n, ...
        # coefs has to be 2D, if empty, shape has to be (0,0)

        self.coefs = coefs
        self.delays = delays

    
    def __call__(self, s: complex | npt.NDArray):
        return self.eval(s)

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
    
    @property
    def antiderivative(self) -> 'QuasiPolynomial':
        raise NotImplementedError(".") # TODO

    def minimal_form(self) -> 'QuasiPolynomial':
        """ Converts QuasiPolynomial to minimal sorted form with no duplicate delays
        """
        coefs, delays = compress(self.coefs, self.delays)
        return QuasiPolynomial(coefs, delays)
    
    @property
    def poly_degrees(self) -> npt.NDArray:
        # TODO is_empty
        return np.apply_along_axis(poly_degree, 1, self.coefs)
    
    def eval(self, s: complex | npt.NDArray):
        return eval(self.coefs, self.delays, s)
    
    def __neg__(self):
        return QuasiPolynomial(-self.coefs, self.delays)
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            coefs, delays = add(
                self.coefs, self.delays,
                np.array([[other]], dtype=self.coefs.dtype), np.array([0.], dtype=self.delays.dtype),
            )
            return QuasiPolynomial(coefs, delays)
        elif isinstance(other, QuasiPolynomial):
            coefs, delays = add(self.coefs, self.delays, other.coefs, other.delays)
            return QuasiPolynomial(coefs, delays)
        else:
            raise ValueError(f"Not possible to add {other}")
    
    def __sub__(self, other):
        return self.__add__(self, -other)
    
    def __pow__(self, other): # TODO
        raise NotImplementedError("")

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return QuasiPolynomial(other * self.coefs, self.delays)
        elif isinstance(other, QuasiPolynomial):
            coefs, delays = multiply(self.coefs, self.delays, other.coefs, other.delays)
            return QuasiPolynomial(coefs, delays)
        else:
            raise ValueError(f"Not possible to multiply with {other}")
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __rsub__(self, other):
        return self.__sub__(other)

    def __rmul__(self, other):
        return self.__mul__(other)
    
    @classmethod
    def constant(cls, const: float, dtype=np.float64):
        """ Creates constant represetation of a quasipolynomial """
        if const == 0: # TODO consider 0 representation? 
            return cls(np.array([[]], dtype=dtype), np.array([], dtype=dtype))
        return cls(np.array([[const]], dtype=dtype), np.array([0], dtype=dtype))
    
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
