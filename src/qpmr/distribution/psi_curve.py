"""
TODO
----

"""

import logging
import numpy as np
import numpy.typing as npt

# from scipy import interpolate

logger = logging.getLogger(__name__)

# coefficients of the functions a(theta) and b(theta) that are used to
# approximate R(N;theta) (which on itself defines the region in which the
# exponential function is sufficiently well approximated). More specifically,
# R(N;theta) ≈ b(theta) + a(theta)*N.

# B_THETA: npt.NDArray = np.array([-0.2462, -0.2266, -0.2053, -0.2302, -0.2326, 
#                                  -0.2335, -0.2362, -0.2421, -0.2463, -0.2604,
#                                  -0.2656, -0.2749, -0.2919, -0.3030, -0.3140,
#                                  -0.3265, -0.3491, -0.3664, -0.3892, -0.4204,
#                                  -0.4548, -0.4855, -0.5339, -0.5872, -0.6491,
#                                  -0.7354, -0.8478, -0.9930, -1.1800, -1.4448,
#                                  -1.9414, -2.7149, -3.0292], dtype=np.float64)
# A_THETA: npt.NDArray = np.array([0.9124, 0.9123, 0.9136, 0.9165, 0.9195, 0.9234,
#                                  0.9285, 0.9345, 0.9416, 0.9501, 0.9592, 0.9698,
#                                  0.9818, 0.9947, 1.0090, 1.0249, 1.0427, 1.0620,
#                                  1.0833, 1.1069, 1.1331, 1.1614, 1.1936, 1.2289,
#                                  1.2685, 1.3132, 1.3642, 1.4231, 1.4913, 1.5731,
#                                  1.6783, 1.7867, 1.8183], dtype=np.float64)
# THETA: npt.NDArray = np.linspace(0, np.pi/2, 33, dtype=np.float64)

# B_THETA_RECT = np.array([-0.2462, -0.2266, -0.2053, -0.2302, -0.2326, -0.2335,
#                          -0.2362, -0.2421, -0.2463, -0.2604, -0.2656, -0.2749,
#                          -0.2919, -0.3030, -0.3140, -0.3265, -0.3491, -0.3664,
#                          -0.3892, -0.4204, -0.4548, -0.4855, -0.5339, -0.5872,
#                          -0.6491, -0.7354, -0.8478, -0.9930, -1.1800, -1.4448,
#                          -1.9414, -2.7149, -3.0292, -2.7272, -2.2329, -1.9425,
#                          -1.7025, -1.5274, -1.3828, -1.2578, -1.1550, -1.0718,
#                          -0.9977, -0.9340, -0.8771, -0.8243, -0.7852, -0.7499,
#                          -0.7131, -0.6879, -0.6574, -0.6386, -0.6108, -0.5970,
#                          -0.5816, -0.5640, -0.5540, -0.5413, -0.5341, -0.5273,
#                          -0.5187, -0.5147, -0.5130, -0.5133, -0.5123])
# A_THETA_RECT = np.array([0.9124, 0.9123, 0.9136, 0.9165, 0.9195, 0.9234, 0.9285,
#                          0.9345, 0.9416, 0.9501, 0.9592, 0.9698, 0.9818, 0.9947,
#                          1.0090, 1.0249, 1.0427, 1.0620, 1.0833, 1.1069, 1.1331,
#                          1.1614, 1.1936, 1.2289, 1.2685, 1.3132, 1.3642, 1.4231,
#                          1.4913, 1.5731, 1.6783, 1.7867, 1.8183, 1.7507, 1.6451,
#                          1.5566, 1.4801, 1.4158, 1.3602, 1.3108, 1.2670, 1.2281,
#                          1.1935, 1.1621, 1.1333, 1.1073, 1.0841, 1.0630, 1.0433,
#                          1.0260, 1.0098, 0.9958, 0.9823, 0.9709, 0.9603, 0.9508,
#                          0.9427, 0.9356, 0.9295, 0.9246, 0.9202, 0.9170, 0.9148,
#                          0.9136, 0.9133])
# THETA_RECT= np.linspace(0, np.pi, 65)

# cubic_spline_a = interpolate.CubicSpline(THETA, A_THETA)
# cubic_spline_b = interpolate.CubicSpline(THETA, B_THETA)

# cubic_spline_a_rect = interpolate.CubicSpline(THETA_RECT, A_THETA_RECT)
# cubic_spline_b_rect = interpolate.CubicSpline(THETA_RECT, B_THETA_RECT)




def psi_comensurate_0_kappa(coefs, n_k, grid_points: int=20):
    """
    TODO
    """
    n, m = coefs.shape

    jhh = np.pi / (grid_points * n_k[-1])
    W = np.zeros(shape=(n-1, n-1), dtype=np.complex128)
    W[1:, :-1] = np.eye(n - 2)

    gk = []
    for k in range(grid_points*n_k[-1]):
        # sigma = np.roots( np.sum( coefs * np.exp(1j*k*jhh*n_k), axis=1) )
        a = np.sum( coefs * np.exp(1j*k*jhh*n_k), axis=1) # [a_0, a_1, ..., a_n=1]
        W[0:, -1] = - a[:-1]
        r, _ = np.linalg.eig(W)
        gk.append(np.conjugate(r))
    
    return np.concatenate(gk)

def psi_comensurate_0_kappa(coefs, n_k, grid_points: int=20):
    """
    TODO
    """
    n, m = coefs.shape

    jhh = np.pi / (grid_points * n_k[-1])
    W = np.zeros(shape=(n-1, n-1), dtype=np.complex128)
    W[1:, :-1] = np.eye(n - 2)

    gk = []
    for k in range(grid_points*n_k[-1]):
        # sigma = np.roots( np.sum( coefs * np.exp(1j*k*jhh*n_k), axis=1) )
        a = np.sum( coefs * np.exp(1j*k*jhh*n_k), axis=1) # [a_0, a_1, ..., a_n=1]
        W[0:, -1] = - a[:-1]
        r, _ = np.linalg.eig(W)
        gk.append(np.conjugate(r))
    
    return np.concatenate(gk)

def psi_commensurate_kappa(coefs, n_k, base_delay, si, grid_points: int=20):
    n, m = coefs.shape
    stepsize = np.pi / grid_points
    factor = 1.05*np.sin(stepsize)
    jhh = np.pi / (grid_points * n_k[-1])
    W = np.zeros(shape=(n-1, n-1), dtype=np.complex128)
    W[1:, :-1] = np.eye(n - 2)

    gk = []
    for k in range(grid_points*n_k[-1]):
        a = np.sum( coefs * np.exp(1j*k*jhh*n_k) * np.exp(-factor * si * base_delay*n_k), axis=1) # [a_0, a_1, ..., a_n=1]
        W[0:, -1] = - a[:-1]
        r, _ = np.linalg.eig(W)
        gk.append(np.conjugate(r))
    
    return np.concatenate(gk)








