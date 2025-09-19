"""
QPmR Info object
----------------

"""

from typing import Callable
from anytree import RenderTree, Node
from functools import cached_property
import contourpy
import numpy as np
import numpy.typing as npt

from qpmr.quasipoly import derivative
from qpmr.quasipoly.core import _eval_array

class QpmrInfo:
    # TODO maybe dataclass? but solve cached property
    real_range: npt.NDArray = None
    imag_range: npt.NDArray = None
    z_value: npt.NDArray = None
    roots0: npt.NDArray = None
    roots_numerical: npt.NDArray = None

    contours_real: list[npt.NDArray] = None
    # contours_imag: list[npt.NDArray] = None

    @cached_property
    def complex_grid(self) -> npt.NDArray:
        return 1j*self.imag_range.reshape(-1, 1) + self.real_range
    
    @cached_property
    def contours_imag(self) -> list[npt.NDArray]:
        contour_generator = contourpy.contour_generator(
            x=self.real_range,
            y=self.imag_range,
            z=np.imag(self.z_value),
        )
        zero_level_contours = contour_generator.lines(0.0)
        return zero_level_contours
    

class QpmrSubInfo(Node):

    region: tuple = None
    ds: float = None
    e: float = None
    status: str = "UNPROCESSED" # TODO enum, unprocessed, failed, solved
    status_message: str = None
    roots: npt.NDArray = None

    z_value: npt.NDArray = None
    roots0: npt.NDArray = None
    roots_numerical: npt.NDArray = None

    contours_real: list[npt.NDArray] = None

    def __init__(self, parent=None):
        self.parent = parent  # This establishes the tree hierarchy

    @property
    def expanded_region(self) -> tuple[float, float, float, float]:
        r = (
            self.region[0] - 3*self.ds,
            self.region[1] + 3*self.ds,
            self.region[2] - 3*self.ds,
            self.region[3] + 3*self.ds,
        )
        return r

    @cached_property
    def real_range(self) -> npt.NDArray:
        bmin, bmax, _, _ = self.expanded_region
        return np.arange(bmin, bmax, self.ds)
    
    @cached_property
    def imag_range(self) -> npt.NDArray:
        _, _, wmin, wmax = self.expanded_region
        return np.arange(wmin, wmax, self.ds)
    
    @cached_property
    def complex_grid(self) -> npt.NDArray:
        return 1j*self.imag_range.reshape(-1, 1) + self.real_range
    
    @cached_property
    def contours_imag(self) -> list[npt.NDArray]:
        contour_generator = contourpy.contour_generator(
            x=self.real_range,
            y=self.imag_range,
            z=np.imag(self.z_value),
        )
        zero_level_contours = contour_generator.lines(0.0)
        return zero_level_contours
    
    @property
    def name(self) -> str:
        return f"QPmR[{self.status}-{self.status_message}] {self.region} ds={self.ds}"
    
class QpmrRecursionContext:
    """ stuff that does not change in recursion + memory for results """
    
    grid_nbytes_max: int = 128_000_000 # 250_000_000
    recursion_level_max: int = 5
    multiplicity_heuristic: bool = False
    
    # numerical method
    numerical_method: str = None # TODO Enum
    numerical_method_kwargs: dict = {}

    ds: float = None

    def __init__(self, coefs, delays):
        self.coefs = coefs
        self.delays = delays
        
        self.solution_tree: QpmrSubInfo = None
        self.node: QpmrSubInfo = None # current node
    
    @property
    def render_tree(self) -> str:
        s = ""
        if self.solution_tree:
            for pre, fill, node in RenderTree(self.solution_tree):
                s += f"{pre}{node.name}\n"
            # TODO delete last two chars
        return s
    
    @property
    def roots(self) -> npt.NDArray:
        if self.solution_tree is None:
            return None
        
        # TODO it can be None if FAILED
        return np.concatenate([leaf.roots for leaf in self.solution_tree.leaves], axis=0)

    @property
    def zeros(self) -> npt.NDArray:
        """ alias for `roots` """
        return self.roots

    @cached_property
    def _qp_prime(self) -> tuple[npt.NDArray, npt.NDArray]:
        return derivative(self.coefs, self.delays)
    
    @property
    def coefs_prime(self) -> npt.NDArray:
        return self._qp_prime[0]

    @property
    def delays_prime(self) -> npt.NDArray:
        return self._qp_prime[1]
    
    @cached_property
    def f(self) -> Callable:
        return lambda s: _eval_array(self.coefs, self.delays, s)

    @cached_property
    def f_prime(self) -> Callable:
        return lambda s: _eval_array(self.coefs_prime, self.delays_prime, s)

