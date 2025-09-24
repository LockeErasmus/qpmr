"""
Quasipolynomial examples
------------------------

This file contains various examples of quasi-polynomials taken from literature.
Some examples are parametrizable, in these cases, calling without passing
arguments is always possible and results in article supplied values. Callables
constructing quasi-polynomials are named using citation key, for the whole
reference and notes read docstring of individual callables.
"""

import numpy as np
import numpy.typing as npt

from .obj import QuasiPolynomial, TransferFunction

def mazanti2021multiplicity(kappa: float=1.964, k: float=-0.67036, tau0: float=0.33, tau1: float=0.33) -> tuple[npt.NDArray, npt.NDArray]:
    """ Example obtained from article [1], for the default setting, there is 
    dominant real pole of multiplicity 6: s = −6.021

    Args:
        kappa (float): parameter from article
        k (float): parameter from article
        tau0 (float): delay from article, should be positive
        tau1 (float): delay from article, should be positive

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

def yuksel2023distributed(return_denum: bool=True) -> tuple[npt.NDArray, npt.NDArray]:
    """ Internal Model Control Transfer function taken from
    
    YUKSEL, Can Kutlu, et al. A distributed delay based controller for
    simultaneous periodic disturbance rejection and input-delay compensation.
    Mechanical Systems and Signal Processing, 2023, 197: 110364.

    Args:
        return_denum (bool): if True, returns quasipolynomial representing
            denumerator of transfer function, set to False to obtain numerator,
            default True

    Returns:
        tuple containing:

            - coefs (array): matrix definition of polynomial coefficients (each row
                represents polynomial coefficients corresponding to delay)
            - delays (array): vector definition of associated delays (each delay
                corresponds to row in `coefs`)
    
    Additional commentary:

    Sensitivity is given by Eq. (12) as

                    1 - C(s) * G_m(s) * exp(-s*tau_m) 
    S(s) = -------------------------------------------------------------
            1 + C(s) * ( G_i(s) * exp(-s*tau) - G_m(s) * exp(-s*tau_m))

    we are interested in showing zeros and poles of S(s).
    
    The model is considered in the form of non delayd LTI as

                    K
        Gm(s) = ---------
                T*s + 1

    Plant Gi(s) was identified as a non delayed LTI of 3rd order and controller
    
    C(s) is given by

                  1
        C(s) = --------- * D(s)
                Gm(s)
    with:
                1
        D(s) = --- (a_0 + a_1 * exp(-s * theta) + ... a_N * exp(-s * N * theta))
                s

    To obtain same results as in article, use:
    
    >>> region = (-12, 1, -0.1, 5000)

    Numerator is expected to have purely imaginary zeros:

    >>> expected_zeros = np.array([4,8,12,16,20,24,28,32]) * 1j * 2 * np.pi
    """
    # Gi - identified system
    Gi_num = np.array([[1.816*1e6, 1.055*1e5, 2286.]])
    Gi_denum = np.array([[3.079*1e6, 2.581*1e5, 9647, 167.5, 1.]])

    # Gm - model parameters K and T
    K = 0.59
    T = 0.018

    # delays
    tau = 0.2
    tau_m = 0.212

    # D(s) - input-shaper like structure
    theta = 0.01
    N = 60
    tau_vector = theta * np.arange(0, N, 1, dtype=np.float64)

    # gains obtained directly from main author
    gains = np.array([3.18419888499517, -8.55006730630735, 7.20804319626175, 20.8242855731446,
                    -6.98819718206270, -20.1849552841814, 2.55098297500236, 5.35346406689096,
                    -5.66728253720343, 1.76092216986347, 2.95360281683304, -4.28799405248918,
                    1.39777992979417, 2.34697134257513, -3.57660291251610, 1.38982228186985,
                    1.98505708058481, -3.46991984257449, 1.62166767533749, 1.89645149324260,
                    -3.86994338042140, 2.11291793462551, 2.18097014111035, -5.13335367003861,
                    2.86163128245455, 3.19992555366216, -8.53434063764039, 7.22376986492874,
                    20.8400122418116, -6.97247051339561, -20.1692286155144, 2.56670964366931,
                    5.36919073555793, -5.65155586853645, 1.77664883853048, 2.96932948550004,
                    -4.27226738382219, 1.41350659846120, 2.36269801124215, -3.56087624384912,
                    1.40554895053682, 2.00078374925181, -3.45419317390750, 1.63739434400443,
                    1.91217816190956, -3.85421671175439, 2.12864460329249, 2.19669680977736,
                    -5.11762700137162, 2.87735795112152, 3.21565222232914, -8.51861396897338,
                    7.23949653359569, 20.8557389104785, -6.95674384472881, -20.1535019468474,
                    2.58243631233637, 5.38491740422488, -5.63582919986946, 1.79237550719740])

    tau = QuasiPolynomial(np.array([[1.]]), np.array([tau]))
    tau_m = QuasiPolynomial(np.array([[1.]]), np.array([tau_m]))
    Gi = TransferFunction(
        num=QuasiPolynomial.from_array(Gi_num),
        denum=QuasiPolynomial.from_array(Gi_denum),
    )
    D = QuasiPolynomial(gains[:, np.newaxis], tau_vector)
    C = TransferFunction(
        num=QuasiPolynomial.from_array(np.array([1, T])), # T*s + 1
        denum=QuasiPolynomial.from_array(np.array([0, K])), # K*s
    ) * D
    Gm = TransferFunction(
        num=QuasiPolynomial.from_array(np.array([K])),
        denum=QuasiPolynomial.from_array(np.array([1, T])),
    )

    tf = (1 - C * Gm * tau_m) / (1 + C * (Gi * tau - Gm * tau_m) )

    if return_denum:
        return tf.denum.coefs, tf.denum.delays
    else:
        return tf.num.coefs, tf.num.delays

def vyhlidal2014qpmr_01() -> tuple[npt.NDArray, npt.NDArray]:
    """ TODO """
    delays = np.array([0, 1.])
    coefs = np.array([[0, 1],[1, 0.]])
    return coefs, delays

def vyhlidal2014qpmr_02() -> tuple[npt.NDArray, npt.NDArray]:
    """ TODO """
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
    return coefs, delays

def vyhlidal2014qpmr_03() -> tuple[npt.NDArray, npt.NDArray]:
    """ TODO """
    delays = np.array([0.0, 1.5, 2.2, 4.3, 6.3])
    coefs = np.array([[2.1, 5, 0, 0.2, 1],
                      [0, -2.1, 0, 0, 0.5],
                      [0, 0, 3.2, 0, 0.3],
                      [0, 0, 1.2, 0, 0,],
                      [3, 0, 0, 0, 0]])
    return coefs, delays

def vyhlidal2014qpmr(example: str | int=1):
    match int(example):
        case 1:
            return vyhlidal2014qpmr_01()
        case 2:
            return vyhlidal2014qpmr_02()
        case 3:
            return vyhlidal2014qpmr_03()
        case _:
            allowed_examples = [1, 2, 3]
            raise ValueError(f"Example '{example}' is not supported. Supported list of examples: {', '.join(allowed_examples)}")

def appeltans2023analysis(example: str=None, **kwargs):
    """ TODO """
    allowed_examples = ["2.6"]
    
    if example not in allowed_examples:
        raise ValueError(f"Example '{example}' is not supported. Supported list of examples: {', '.join(allowed_examples)}")
    
    if example == "2.6":
        tau = kwargs.get("tau2", 2.0)
        coefs = np.array([[-1/4, 1.],
                          [1/3, -3/4],
                          [0, 1/2]])
        delays = np.array([0., 1, tau], dtype=float)
    
    return coefs, delays

def self_inverse_polynomial(center: float=0., radius: float=1.0, degree: int=6):
    coefs = ( (1 / radius * np.poly1d([1, -center])) ** degree - 1 ).coeffs[None, ::-1]
    delays = np.array([0.])
    return coefs, delays

def empty():
    """ Constructs emtpy quasi-polynomial representation -> h(s) = 0 """
    return np.array([[]], dtype=np.float64), np.array([], dtype=np.float64)

