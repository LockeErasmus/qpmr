"""

"""

import logging

import numpy as np
import numpy.typing as npt
import numba

logger = logging.getLogger()

def create_callable(coefs: npt.NDArray, delays: npt.NDArray, powers: npt.NDArray=None, **kwargs):
    """

    Args:
        coefs
        delays
        powers (array): 1d array of powers, if None powers are assumed to be
            [0, 1, ..., d+1]
        **kwargs:
            njit_kwargs (dict): kwargs for njit decorator, default {'nopython': True}
    """

    # resolve kwargs
    njit_kwargs = kwargs.get("njit_kwargs", dict(nopython=True))

    # perform QP minimalisation TODO


    if not powers:
        # no powers -> simple quasi-polynomial
        @numba.jit(**njit_kwargs)
        def f(z: npt.NDArray) -> npt.NDArray:
            val = np.sum(np.multiply(
                np.sum(
                    (np.power(z[..., np.newaxis], powers[np.newaxis, :])[..., np.newaxis]
                    * coefs.T[np.newaxis, ...]),
                    axis=-2
                ),
                np.exp(z[..., np.newaxis] * - delays[np.newaxis, ...]),
            ), axis=-1)
            return val

    else:
        # powers defined -> fractional order quasi-polynomial

        # nopython, parallel, ...
        @numba.jit(**njit_kwargs)
        def f(z: npt.NDArray) -> npt.NDArray:
            val = np.sum(np.multiply(
                np.sum(
                    (np.power(z[..., np.newaxis], powers[np.newaxis, :])[..., np.newaxis]
                    * coefs.T[np.newaxis, ...]),
                    axis=-2
                ),
                np.exp(z[..., np.newaxis] * - delays[np.newaxis, ...]),
            ), axis=-1)
            return val
        

    df = None
    return f, df



if __name__ == "__main__":
    delays = np.array([24.99, 23.35, 19.9, 18.52, 13.32, 10.33, 8.52, 4.61, 0.0])
    coefs = np.array([[51.7, 0, 0, 0, 0, 0, 0, 0 , 0],
                        [1.5, -0.1, 0.04, 0.03, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0.5, 0, 0, 0, 0, 0],
                        [0, 25.2, 0, -0.9, 0.2, 0.15, 0, 0, 0],
                        [7.2, -1.4, 0, 0, 0.1, 0, 0.8, 0, 0],
                        [0, 19.3, 2.1, 0, -8.7, 0, 0, 0, 0],
                        [0, 6.7, 0, 0, 0, -1.1, 0, 1, 0],
                        [29.1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, -1.8, 0.001, 0, 0, -12.8, 0, 1.7, 0.2]])
    

    f, df = create_callable(delays, coefs)

    region = [-1, 20, 0, 20]
    ds = 0.01
    bmin=region[0] - 3*ds
    bmax=region[1] + 3*ds
    wmin=region[2] - 3*ds
    wmax=region[3] + 3*ds

    real_range = np.arange(bmin, bmax, ds)
    imag_range = np.arange(wmin, wmax, ds)
    grid = 1j*imag_range.reshape(-1, 1) + real_range

    Z = f(grid)

