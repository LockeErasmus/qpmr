"""
DDAE implementation
"""

import numpy as np
import numpy.typing as npt

"""

Steps:
    1. sort delays and matrices, remove repetitions

"""

class DDAE:

    def __init__(self, A: list[npt.NDArray], hA: list[float], E=None, **kwargs) -> None:
        
        assert len(A) >= 1, "At least one matrix of dynamics is required"
        assert len(A) == len(hA)
        assert all(delay>=0 for delay in hA), "Positive delay, advanced TDS not allowed"
        
        
        assert np.atleast_2d(*A)
        shape = A[0]
        assert all(a.shape == shape for a in A), "A_i Matrices have to have same shape"
        assert all(a.shape[0] == a.shape[1] for a in A), "A_i matrices have to be square"

        
        
        # ---
        self._A = A
        self._hA = hA

        self._E = E

        self._n = shape[0]

        # ---
        self.dtype = kwargs.get("dtype", np.float64)


        @property
        def n(self) -> int:
            return self._n

        @property
        def E(self) -> npt.NDArray:
            if self._E is None:
                return np.eye(self.n, dtype=self.dtype)
            else:
                return self._E

        pass