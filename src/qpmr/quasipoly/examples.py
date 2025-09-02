"""
Quasipolynomial examples
------------------------

Notes:
    1. quasipolynomials from articles are named as a code from bibtex

"""

import numpy as np
import numpy.typing as npt

def mazanti2021multiplicity(kappa: float=1.964, k: float=-0.67036, tau0: float=0.33, tau1: float=0.33) -> tuple[npt.NDArray, npt.NDArray]:
    """ Example obtained from article [1], for the default setting, there is 
    dominant real pole of multiplicity 6: s = −6.021

    Args:
        kappa (float)
        k (float)
        tau0 (float): should be positive
        tau1 (float): should be positive

    Returns:
        tuple containing:

            - coefs (array): matrix definition of polynomial coefficients (each row
                represents polynomial coefficients corresponding to delay)
            - delays (array): vector definition of associated delays (each delay
                corresponds to row in `coefs`)
    
    
    [1] MAZANTI, Guilherme; BOUSSAADA, Islam; NICULESCU, Silviu-Iulian.
    Multiplicity-induced-dominancy for delay-differential equations of retarded
    type. Journal of Differential Equations, 2021, 286: 84-118.

    Notes:
        - Parameters calculated corresponding to Proposition 6.1,
          Eq. (6.10a) - (6.10f), see notes in code
        - have in mind that due to rounding, it is expected that the poles are
          distributed on the disk around -6.021
    """
    tau = tau0 + tau1
    r0 = -3. - (9)**(1/3.) + (3)**(1/3.)
    s0 = r0 / tau - 1 / kappa # (6.10a)
    omega = (-kappa * (s0**3 + 9./tau *s0**2 + 36./tau/tau * s0 +60/tau**3) )**(1./2) # (6.10b)
    xi = -3.*s0/2/omega - 9./2/omega/tau - 1/2/omega/kappa # (6.10c)
    beta0 = - (3*kappa*(s0**2*tau**2 - 8*s0*tau + 20) * np.exp(s0*tau) ) / (k*omega**2*tau**3) # (6.10d)
    beta1 = (6* kappa * (s0*tau - 4) * np.exp(s0*tau)) / (k*omega**2*tau**2) # (6.10e)
    beta2 = (-3*kappa * np.exp(s0*tau)) / (k*omega**2*tau) # (6.10f)
    
    # quasipolynomial as coefs, delays
    coefs = np.array([
        [omega**2/kappa, omega**2 + 2*omega*xi/kappa, (2*omega*xi + 1/kappa), 1.],
        [-beta0*k*omega**2/kappa, -beta1*k*omega**2/kappa, -beta2*k*omega**2/kappa, 0.]
    ])
    delays = np.array([0., tau])

    return coefs, delays


def self_inverse_polynomial(center: float=0., radius: float=1.0, degree: int=6):
    coefs = ( (1 / radius * np.poly1d([1, -center])) ** degree - 1 ).coeffs[None, ::-1]
    delays = np.array([0.])
    return coefs, delays
