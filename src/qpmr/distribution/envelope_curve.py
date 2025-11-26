"""
Envelope curve
--------------
TODO
"""
import itertools
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
                n = len(delays) - 1
                A0 = np.zeros(shape=(n, n))
                A0[-1,:] = -coefs[k,:-1]
                A0[:-1, 1:] = np.eye(n-1)
                # roots = np.roots(coefs[k,:-1][::-1])
                norms[k] = np.linalg.norm(A0, ord=2)
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
    return np.sqrt( np.square(np.sum(np.exp(-real[:, None] * delays[None, :]) * norms[None, :], axis=1)) - np.square(real) )



def _predictor(p, delays, x: float):

    # p = [1, p1, p2, ...]
    # delays = [0, tau1, tau2, ...]

    # form problem as minimization
    # maximize 1 / |f(\theta)| => minimize |f(\theta)|
    
    n_theta = len(delays) - 1
    
    n = 20 # discretization
    theta_discretization = np.linspace(0, np.pi, n) # the problem is symetrical

    vec1 = p * np.exp(-x*delays)
    val_star = np.inf
    for th in itertools.product(theta_discretization, repeat=n_theta):
        vec2 = np.exp(1j * np.array(th, dtype=np.complex128))
        val = np.abs(1 - np.inner(vec1, vec2))
        if val < val_star:
            val_star = val
    
    return
    


def _predictor_commensurate():
    raise NotImplementedError


def envelope_real_axis_crossing(coefs, delays):
    """
    """

    # TODO: coefs and delays are compressed and normalized and neutral or retarded
    pass