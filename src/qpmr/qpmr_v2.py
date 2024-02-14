"""
QPmR v2 implementation
"""
from functools import cached_property
import logging

import contourpy
import numpy as np
import numpy.typing as npt

from .numerical_methods import numerical_newton
from .argument_principle import argument_principle

logger = logging.getLogger(__name__)

def grid_size_heuristic(region, *args, **kwargs) -> float:
    r = (region[1] - region[0]) * (region[3] - region[2]) / 1000.
    return r

def find_roots(x, y) -> npt.NDArray:
    """ Finds 0-level crossings by checking consequent difference in signs, ie
    
    s[k] is True if sign(y[k]), sign(y[k+1]) is either `+-` or `-+`

    removing duplicates: if `+-+` or `-+-` sequence is present -> only first occurence

    """
    s = np.abs(np.diff(np.sign(y))).astype(bool)
    duplicates = s[:-1] & s[1:] # duplicates mask
    s[1:] = np.bitwise_xor(s[1:], duplicates) # removes duplicates
    return x[:-1][s] + np.diff(x)[s]/(np.abs(y[1:][s]/y[:-1][s])+1)

def create_vector_callable(coefs, delays):
    degree, num_delays = np.shape(coefs)
    def func(z):
        shape = np.shape(z)
        _memory = np.ones(shape, dtype=z.dtype)
        delay_terms = np.exp(
            (np.tile(z, (len(delays), 1))
             * (-delays[:, np.newaxis]))
        )
        val = np.zeros(shape, dtype=z.dtype)
        for d in range(degree):
            val += np.sum(delay_terms * _memory[np.newaxis, :] * coefs[:, d][:, np.newaxis], axis=0)
            _memory *= z
        return val
    return func


class QpmrOutputMetadata:
    # TODO maybe dataclass? but solve cached property
    real_range: npt.NDArray = None
    imag_range: npt.NDArray = None
    z_value: npt.NDArray = None

    contours_real: list[npt.NDArray] = None
    contours_imag: list[npt.NDArray] = None

    @cached_property
    def complex_grid(self) -> npt.NDArray:
        return 1j*self.imag_range.reshape(-1, 1) + self.real_range

def qpmr(
        region: list[float, float, float, float],
        coefs: npt.NDArray,
        delays: npt.NDArray,
        **kwargs) -> tuple[npt.NDArray | None, QpmrOutputMetadata]:
    """

    Args:
        region: TODO
        coefs: TODO
        delays: TODO

        **kwargs:
            e (float) - computation accuracy, default = 1e-6
            ds (float) - grid step, default obtained by heuristic

            newton_max_iterations: int
    
    """

    assert len(region) == 4, "region is expected to be of a form [Re_min, Re_max, Im_min, Im_max]"
    assert region[0] < region[1], f"region boundaries on real axis has to fullfill {region[0]} < {region[1]}"
    assert region[2] < region[3], f"region boundaries on imaginary axis has to fullfill {region[2]} < {region[3]}"

    # TODO assert coefs dimensions, match delays dimensions
    # TODO assert degree >= 0, meaning coefs has at least 1 column
    
    # defaults
    e = kwargs.get("e", 1e-6)
    ds = kwargs.get("ds", None)
    if not ds:
        ds = grid_size_heuristic(region)
        logger.debug(f"Grid size not specified, setting as ds={ds} (solved by heuristic)")
    
    # roots precission - newtom method
    newton_max_iterations = kwargs.get("newton_max_iterations", 100)
    # TODO add others as well
    assert isinstance(e, float) and e > 0.0, "error 'e' numerical accuracy"
    assert isinstance(ds, float) and ds > 0.0, "error 'ds' grid stepsize"

    metadata = QpmrOutputMetadata()

    # extend region and create meshgrid (original algorithm) -> TODO move to function
    bmin=region[0] - 3*ds
    bmax=region[1] + 3*ds
    wmin=region[2] - 3*ds
    wmax=region[3] + 3*ds
    real_range = np.arange(bmin, bmax, ds)
    imag_range = np.arange(wmin, wmax, ds)
    # add to metadata
    metadata.real_range = real_range
    metadata.imag_range = imag_range
    complex_grid = metadata.complex_grid # 1j*imag_range.reshape(-1, 1) + real_range
    
    # values of function -> TODO move to separate function
    degree, num_delays = coefs.shape # TODO variables keep, move up and rework
    func_value = np.zeros(complex_grid.shape, dtype=complex_grid.dtype)
    _memory = np.ones(complex_grid.shape, dtype=complex_grid.dtype) # x*x*..*x much faster than np.power(x, N)
    ## prepare exp(-s*tau)
    # TODO delays has to be (num_delays,) vector
    delay_grid = np.exp(
        (np.tile(complex_grid, (len(delays), 1, 1)) # N times complex grid, [0,:,:] is complex grid for delay1 etc.
         * (-delays[:, np.newaxis, np.newaxis])) # power of broadcasting [delay1, delay2, ..., delayN]
    )
    for d in range(degree):
        func_value += np.sum(delay_grid * _memory[np.newaxis, :, :] * coefs[:, d][:, np.newaxis, np.newaxis], axis=0)
        _memory *= complex_grid 

    metadata.z_value = func_value

    func_value_real = np.real(func_value)
    # func_value_imag = np.imag(func_value) # not needed in original algorithm

    ## finding contours via contourpy library
    contour_generator = contourpy.contour_generator(x=real_range, y=imag_range, z=func_value_real) # TODO other kwargs can go here
    zero_level_contours = contour_generator.lines(0.0) # find all 0 level curves

    if not zero_level_contours: # list is empty, i.e []
        logger.warning(f"No real 0-level contours were found in region {region}.")
        return None, metadata
    
    # detecting intersection points 
    roots = []
    for polygon in zero_level_contours:
        polygon_complex = polygon[:,0] + 1j*polygon[:,1]
        
        # calculate value for all of these points
        delay_terms = np.exp(
            (np.tile(polygon_complex, (len(delays), 1))
             * (-delays[:, np.newaxis]))
        )
        polygon_func_value = np.zeros(polygon_complex.shape, dtype=polygon_complex.dtype)
        _memory = np.ones(polygon_complex.shape, dtype=polygon_complex.dtype)
        for d in range(degree):       
            polygon_func_value += np.sum(delay_terms * _memory[np.newaxis, :] * coefs[:, d][:, np.newaxis], axis=0)
            _memory *= polygon_complex # _memory = np.multiply(_memory, complex_grid)
        
        # find all intersections
        polygon_func_imag = np.imag(polygon_func_value)
        crossings = find_roots(polygon_complex, polygon_func_imag)
        if crossings.size:
            roots.append(crossings)
    
    if not roots: # no crossings found
        logger.warning(f"No crossings contour crossings found!") # TODO better message
        return None, metadata
    
    roots0 = np.hstack(roots)

    # apply numerical method to increase precission
    func = create_vector_callable(coefs, delays)
    roots = numerical_newton(func, roots0) # TODO inplace=True?
    if False: # if newton did not converge
        ... # run QPmR with ds=ds/3

    # filter out roots that are not in predefined region
    mask = ((roots.real >= region[0]) & (roots.real <= region[1]) # Re bounds
            & (roots.imag >= region[2]) & (roots.imag <= region[3])) # Im bounds
    roots = roots[mask]

    # TODO case where roots found, but are outside of defined region

    # TODO check the distance from the first approximation of the roots is
    # less then 2*ds - as matlab line 629
    dist = np.abs(roots - roots0[mask])
    num_dist_violations = (dist > 2*ds).sum()
    if num_dist_violations > 0:
        logger.warning("2*delta s violated") # TODO message
        # TODO follow-up behaviour
        # run QPmR with ds=ds/3
    
    # TODO argument check - as matlab line 651
    # implement separate function - as matlab line 1009
    n = argument_principle(func, region, ds/10., eps=e/100.) # ds and eps obtained from original matlab implementation
    smaller_region = [region[0]+ds/10., region[1]-ds/10., region[2]+ds/10.,region[3]-ds/10.]
    n_smaller = argument_principle(func, smaller_region, ds/10., eps=e/100.)
    if len(roots) == n or len(roots) == n_smaller:
        # ok, continue
        pass
    else:
        logger.info(f"Argument principle: {n}, real number of roots {len(roots)}")
        logger.info(f"Argument principle (smaller Region): {n_smaller}, real number of roots {len(roots)}")
        # TODO follow-up behaviour
        # run QPmR with ds=ds/3
        modified_kwargs = kwargs.copy()
        modified_kwargs['ds'] = ds / 3.0
        return qpmr(region, coefs, delays, **modified_kwargs)



    _roots_str = '    \n'.join([str(r) for r in roots])
    #logger.info(f"roots: {_roots_str}")
    return roots, metadata
