"""
Envelope curve
--------------
TODO
"""

import numpy as np
import numpy.typing as npt

from qpmr.numerical_methods import newton

def _spectral_norms(coefs: npt.NDArray, delays: npt.NDArray) -> npt.NDArray:
    """ Calculate spectral norm associated with polynomials
    
    Args:
        coefs (array): matrix definition of polynomial coefficients (each row
            represents polynomial coefficients corresponding to delay)
        delays (array): vector definition of associated delays (each delay
            corresponds to row in `coefs`)
    
    Returns:
        norms (array): vector of spectral norms associated to polynomials

    Notes:
        1. assumes quasipolynomial is in minimal, sorted and normalized form
    """
    norms = np.zeros_like(delays, dtype=float)
    for k in range(len(delays)):
        if k == 0:
            monic_coefs = coefs[k,:-1][::-1] # without leading 1
            if len(monic_coefs) == 0:
                norms[k] = 0
            elif len(monic_coefs) == 1:
                norms[k] = abs(monic_coefs[0])
            else:
                roots = np.roots(coefs[k,:-1][::-1])
                norms[k] = np.max(np.abs(roots))                
        else:
            norms[k] = np.linalg.norm(coefs[k,:-1], ord=2)
    return norms

def _envelope_real_axis_crossing(spectral_norms: npt.NDArray, delays: npt.NDArray, **newton_kwargs):
    """
    TODO

    """
    f = lambda x: x - np.inner(spectral_norms, np.exp(-x*delays))
    f_prime = lambda x: 1 - np.inner(-delays * spectral_norms, np.exp(-x*delays))
    x_star, converged = newton(f, f_prime, x0=0.0, **newton_kwargs)
    
    if converged:
        return x_star
    
    raise ValueError(f"Newton did not converge")

def _envelope_imag_axis_crossing(spectral_norms: npt.NDArray) -> float:
    """ symetrical by real axis of course """
    return np.sum(spectral_norms)

def _envelope_eval(real: npt.NDArray, norms: npt.NDArray, delays: npt.NDArray):


    # r = np.sum(np.exp(-np.real(complex_grid)[:,:,None]*delays[None,:])*alphas, axis=-1) - np.abs(complex_grid)
    raise NotImplementedError


