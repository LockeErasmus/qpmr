"""
Example: MID (Multiplicity induced dominancy)
---------------------------------------------
See:

MAZANTI, Guilherme; BOUSSAADA, Islam; NICULESCU, Silviu-Iulian.
Multiplicity-induced-dominancy for delay-differential equations of retarded
type. Journal of Differential Equations, 2021, 286: 84-118.

for quasipolynomial see Equation (6.9)
"""

import numpy as np
import qpmr

kappa = 1.964 # [s]
k = -0.67036 # [rad^{-1}]
tau0 = 0.33 # [s]
tau1 = 0.33 # [s], tau1=0.70

# Parameters calculated corresponding to Proposition 6.1, Eq. (6.10a) - (6.10f)
tau = tau0 + tau1
r0 = -3. - (9)**(1/3.) + (3)**(1/3.)
s0 = r0 / tau - 1 / kappa # (6.10a)
omega = (-kappa * (s0**3 + 9./tau *s0**2 + 36./tau/tau * s0 +60/tau**3) )**(1./2) # (6.10b)
xi = -3.*s0/2/omega - 9./2/omega/tau - 1/2/omega/kappa # (6.10c)
beta0 = - (3*kappa*(s0**2*tau**2 - 8*s0*tau + 20) * np.exp(s0*tau) ) / (k*omega**2*tau**3) # (6.10d)
beta1 = (6* kappa * (s0*tau - 4) * np.exp(s0*tau)) / (k*omega**2*tau**2) # (6.10e)
beta2 = (-3*kappa * np.exp(s0*tau)) / (k*omega**2*tau) # (6.10f)

print(f"{r0=} {s0=} {omega=} {xi=} {beta0=} {beta1=} {beta2=}")

# quasipolynomial as coefs, delays
coefs = np.array([
    [omega**2/kappa, omega**2 + 2*omega*xi/kappa, (2*omega*xi + 1/kappa), 1.],
    [-beta0*k*omega**2/kappa, -beta1*k*omega**2/kappa, -beta2*k*omega**2/kappa, 0.]
])
delays = np.array([0., tau])
region = (-15, 1, 0, 50)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import qpmr
    import qpmr.plot

    from qpmr.qpmr_v3 import qpmr as qpmr_v3

    logger = qpmr.init_logger(level="DEBUG")

    roots, meta = qpmr_v3(region, coefs, delays)
    complex_grid = meta.complex_grid
    value = meta.z_value
    
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(8,5))
    qpmr.plot.qpmr_contour(roots, meta, ax=ax1)
    qpmr.plot.roots(roots, ax=ax2)
    plt.show()