"""
Test functionality connected to spectrum envelope calculation
"""

import numpy as np
import numpy.typing as npt
import pytest

import qpmr.quasipoly.examples as examples
from qpmr.quasipoly import eval, derivative, compress
from qpmr.distribution.psi_curve import psi_comensurate_0_kappa, psi_commensurate_kappa
from qpmr.distribution.envelope_curve import _spectral_norms, _envelope_real_axis_crossing, _envelope_imag_axis_crossing, _envelope_eval
from qpmr.qpmr_v3 import qpmr as qpmr_v3
from qpmr.qpmr_v3 import QpmrSubInfo, QpmrRecursionContext
import qpmr.plot

@pytest.mark.parametrize(
    argnames="qp, base_delay, params",
    argvalues=[
        (examples.vyhlidal2014qpmr_02(), 0.1, {"grid_points": 1}),
        (examples.vyhlidal2014qpmr_03(), 0.01, {"grid_points": 10}),
    ],
    ids=[
        "vyhlidal2014qpmr-02",
        "vyhlidal2014qpmr-03",
    ],
)
def test_psi_commensurate(qp, base_delay, params: dict, enable_plot: bool):    
    
    coefs, delays = qp # unpack quasipolynomial
    coefs, delays = compress(coefs, delays)
    coefs /= coefs[0, -1] # normalize

    # envelope
    norms = _spectral_norms(coefs, delays)
    re_star = _envelope_real_axis_crossing(norms, delays)
    im_star = _envelope_imag_axis_crossing(norms)

    # psi curves
    n_k = np.round(delays / base_delay, decimals=0).astype(int)

    gk = psi_comensurate_0_kappa(coefs, n_k, grid_points=params.get("grid_pints", 20))
    gk = gk[np.isfinite(gk)] # get rid of inf and NaN
    si = np.max(np.real(gk))
    stepsize = np.pi / params.get("grid_pints", 20)
    factor = 1.05*np.sin(stepsize)
    points_psi_0_kappa = gk[(np.real(gk) >= 0) & (np.real(gk) <= factor * si)]
    #theta_points1 = np.abs(np.angle(points_psi_0_kappa))
    #r_points1 = np.abs(points_psi_0_kappa)

    gk2 = psi_commensurate_kappa(coefs, n_k, base_delay, si, grid_points=params.get("grid_pints", 20))
    gk2 = gk2[np.isfinite(gk2)] # get rid of inf and NaN
    points_psi_kappa = gk2[np.real(gk2) >= factor * si]
    #theta_points2 = np.abs(np.angle(points_psi_kappa))
    #r_points2 = np.abs(points_psi_kappa)


    if enable_plot:
        import matplotlib.pyplot as plt
        import contourpy

        fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2,2)
        ax1.scatter(gk.reshape(-1).real, gk.reshape(-1).imag, alpha=0.3, color="red")
        ax2.scatter(points_psi_0_kappa.reshape(-1).real, points_psi_0_kappa.reshape(-1).imag, alpha=0.3, color="red")
        
        ax3.scatter(gk2.reshape(-1).real, gk2.reshape(-1).imag, alpha=0.3, color="blue")
        ax4.scatter(points_psi_kappa.reshape(-1).real, points_psi_kappa.reshape(-1).imag, alpha=0.3, color="blue")
        
        fig, ax = plt.subplots()
        ax.scatter(points_psi_0_kappa.reshape(-1).real, points_psi_0_kappa.reshape(-1).imag, alpha=0.3, color="red")
        ax.scatter(points_psi_kappa.reshape(-1).real, points_psi_kappa.reshape(-1).imag, alpha=0.3, color="blue")

        # plot envelope
        # real_range = np.arange(-2, re_star+5, 0.01)
        # imag_range = np.arange(-im_star*1.1+1, im_star-1.1+1, 0.01)
        # complex_grid = 1j*imag_range.reshape(-1, 1) + real_range
        # r = np.sum(np.exp(-np.real(complex_grid)[:,:,None]*delays[None,:])*norms, axis=-1) - np.abs(complex_grid)
        # contour_generator = contourpy.contour_generator(x=real_range, y=imag_range, z=r)
        # zero_level_contours = contour_generator.lines(0.0) # find all 0-level real contours

        # ax.axhline(0, alpha=0.5, color="black", linestyle="-.")
        # ax.axvline(0, alpha=0.5, color="black", linestyle="-.")

        # for i, c in enumerate(zero_level_contours): # there should be only one, but plot all
        #     ax.plot(c[:,0], c[:,1], color="b", alpha=0.5, label=f"envelope (0-contour ix:{i})")
        x = np.linspace(0, re_star, 1000)
        y = _envelope_eval(x, norms, delays)
        ax.plot(x,y,label="envelope")
        ax.axvline(re_star, color="r", alpha=0.5, label="re*")
        ax.plot([0,0], [-im_star, im_star], marker="o", color="r", alpha=0.5, label="im*")

        #ax.scatter(roots.real, roots.imag, alpha=1.0, color="black")

        plt.show()



