"""
Accuracy of mapping algorithm
-----------------------------
"""

import numpy as np
import scipy.special
import matplotlib.pyplot as plt

from qpmr.core import spectrum_mapping
import qpmr.quasipoly.examples as examples

import settings

# Example
coefs, delays = examples.vyhlidal2014qpmr(1)
region = (-10, 2, 0, 30)

# We can obtain analytically roots for this example
roots_analytical = scipy.special.lambertw(-1, k=np.array([0,1,2,3,4]), tol=1e-12)

region = (-10, 2, 0, 100)
roots_analytical = scipy.special.lambertw(-1, k=np.arange(0, 16, 1), tol=1e-12)
# coefs, delays = examples.vyhlidal2014qpmr(3)
# region = (-2, 7, 0, 9)

plt.figure()

print(delays)
print(coefs)

Nre = []
Nim = []

roots_evolution = []
ds_vector = np.linspace(0.01*np.pi/np.max(delays), 0.6*np.pi/np.max(delays), 1000)[::-1]

for ds in ds_vector:
    # print(len(np.arange(region[0], region[1], ds)),
    #       len(np.arange(region[2], region[3], ds)) )    
    roots = spectrum_mapping(coefs, delays, region, ds=ds)
    Nre.append( int( (region[1] - region[0]) / ds) + 1 )
    Nim.append( int( (region[3] - region[2]) / ds) + 1 )
    roots_evolution.append(roots)

    plt.scatter(np.real(roots), np.imag(roots), alpha=0.2, marker="+")

plt.scatter(np.real(roots_analytical), np.imag(roots_analytical), alpha=0.8, marker="x", color="red")

roots_evolution = np.array(roots_evolution, dtype=np.complex128)
print(roots_evolution.shape)

# Compute relative error of the roots
# error is 

rel_errors = np.divide(
    np.abs(roots_evolution - roots_analytical),
    np.abs(roots_evolution) # +1e-12 # to avoid division by zero
)
abs_errors = np.abs(roots_evolution - roots_analytical)

# plt.figure()
# plt.plot(ds_vector, rel_errors[:, 0])
# plt.axvlinïn(0.1 * np.pi / np.max(delays), color="red", linestyle="--",  label="0.1 * pi / tau_max")
# plt.xlabel("ds")
# plt.ylabel("Relative error of the root")
# plt.legend()

# plt.figure()
# plt.loglog(ds_vector, abs_errors[:, 0], label="Absolute error of the root")
# plt.loglog(ds_vector, np.sqrt(2.) * ds_vector, label=r"$\sqrt{2}d_s$ - theoretical error bound") # error of the half-interval method
# plt.loglog(ds_vector, np.sqrt(3./2) * ds_vector, label=r"$\sqrt{\frac{3}{2}}d_s$ - 2nd theoretical error bound") # error of the half-interval method
# #plt.semilogy(ds_vector, abs_errors[:, 0])
# #plt.semilogy(ds_vector, np.sqrt(2) * ds_vector, label=r"$\sqrt{2} * ds$ - theoretical error bound") # error of the half-interval method
# plt.axvline(0.1 * np.pi / np.max(delays), color="red", linestyle="--", label="0.1 * pi / tau_max")
# plt.xlabel("ds")
# plt.ylabel("Absolute error of the root")
# plt.legend()

fig = plt.figure(figsize=(settings.LINE_WIDTH, settings.MEDIUM_HEIGHT))
for i in range(abs_errors.shape[1]):
    plt.semilogy(ds_vector, abs_errors[:, i], color="blue", alpha=0.1, zorder=1, linewidth=0.5)

plt.semilogy(ds_vector, abs_errors[:, i], color="blue", alpha=0.9, zorder=2, linewidth=1)
plt.semilogy(ds_vector, np.max(abs_errors, axis=1), color="red", zorder=3, linewidth=1)
plt.semilogy(ds_vector, np.sqrt(2.) * ds_vector, "--", color="green", zorder=3, linewidth=1)
plt.axvline(0.1 * np.pi / np.max(delays), color="black", linestyle="--", alpha=0.5, zorder=2, linewidth=1)
plt.xlabel(r"$d_s$")
plt.ylabel(r"$|s_k - W_k(-1)|$")
plt.xlim(ds_vector[-1], ds_vector[0])
# plt.grid(True, which="both", linestyle="--", linewidth=0.5)

settings.save_figure(fig, "accuracy_abs_error")

# relative error



# for i in range(1, len(roots_analytical)):
#     plt.figure()
#     # plt.plot([x*y for x,y in zip(Nre, Nim)])

#     plt.plot(np.real(roots_evolution[:, i]), np.imag(roots_evolution[:, i]), marker="o", color="blue", markersize=2, alpha=0.2, zorder=2)
#     # plt.plot(np.real(roots_evolution[:, 1]), np.imag(roots_evolution[:, 1]), marker="o", color="blue", markersize=2, alpha=0.5)
#     # plt.scatter([np.real(roots_analytical[0])], [np.imag(roots_analytical[0])], marker="x", color="red", s=5, label=r"$W__{0}(-1)$")
#     plt.scatter(np.real(roots_evolution[:1, i]), np.imag(roots_evolution[:1, i]), marker="o", color="green", s=20, alpha=0.8, label=r"max ds", zorder=3)
#     plt.scatter(np.real(roots_analytical[i:i+1]), np.imag(roots_analytical[i:i+1]), marker="o", color="red", s=20, alpha=0.8, label=r"$W_k(-1)$", zorder=3)
#     plt.legend()

plt.show()





