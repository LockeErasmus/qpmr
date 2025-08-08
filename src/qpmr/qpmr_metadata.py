"""
QPmR Info object
----------------

"""

from functools import cached_property
import contourpy
import numpy as np
import numpy.typing as npt

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