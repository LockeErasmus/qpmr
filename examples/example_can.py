"""

Sensitivity is given by equation (12) as

                    1 - C(s) * G_m(s) * exp(-s*tau_m) 
    S(s) = -------------------------------------------------------------
            1 + C(s) * ( G_i(s) * exp(-s*tau) - G_m(s) * exp(-s*tau_m))

with:
            1
    D(s) = --- (a_0 + a_1 * exp(-s * theta) + ... a_N * exp(-s * N * theta))
            s

and:

               K
    G(s) = --------- * exp(-s*tau_m)
            T*s + 1


In this example, we are interested in the location of zeros of the input shaper
with the following Laplace transform:

                    1
    S(s) = gamma + --- (a_0 + a_1 * exp(-s*tau_1) + ... + a_N * exp(-s*tau_N))
                    s


"""

import numpy as np

gamma = 0.25
a_vector = np.array([
    0.0021, 0.5170, 1.1402, 1.3194, 1.0347, 0.4305, -0.2457, -0.7440, -0.9116,
    -0.7504, -0.4075, -0.0992, -0.0004, -0.1478, -0.4044, -0.5105, -0.2114,
    -0.0108, 0.0029, -0.0031
])
tau_vector = np.linspace(0, 0.45, 20)

s0 = -2.3844 + 1j* 23.7554

region = (-100, 0, 0, 100)



# Gi
Gi_num = np.array([[1.816*1e6, 1.055*1e5, 2286.]])
Gi_denum = np.array([[3.079*1e6, 2.581*1e5, 9647, 167.5, 1.]])

# D(s)
theta = 0.01
N = 59
tau_vector = theta * np.arange(0, N+1, 1, dtype=np.float64)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import qpmr
    import qpmr.plot

    logger = qpmr.init_logger(level="DEBUG")

    # coefs
    vec = np.full_like(a_vector, fill_value=0.0)
    vec[0] = gamma

    coefs = np.c_[a_vector, vec]
    delays = tau_vector

    roots, meta = qpmr.qpmr(region, coefs, delays)
    complex_grid = meta.complex_grid
    value = meta.z_value
    
    fig, ax = plt.subplots(1,1,figsize=(12,6))
    qpmr.plot.roots_basic(roots, ax=ax)
    plt.show()



