"""
WIP - Region heuristic 
"""

import qpmr.quasipoly.examples as examples
from qpmr.region_heuristic import region_heuristic
from qpmr.qpmr_v3 import qpmr as qpmr_v3
from qpmr.qpmr_v3 import QpmrSubInfo, QpmrRecursionContext
from qpmr.distribution.zero_chains import chain_asymptotes, chain_asymptotes2
from qpmr.argument_principle import argument_principle_rectangle
import qpmr.quasipoly
import matplotlib.pyplot as plt

from qpmr.distribution.envelope_curve import _spectral_norms, _envelope_real_axis_crossing, _envelope_eval
import qpmr.plot

if __name__ == "__main__":
    
    qpmr.init_logger(level="DEBUG")

    import numpy as np
    
    coefs, delays = examples.vyhlidal2014qpmr(example=2)
    region=(-10, 3, 0, 80)

    coefs, delays = qpmr.quasipoly.compress(coefs, delays)

    # mi_vec, abs_omega = chain_asymptotes(coefs, delays)
    norms = _spectral_norms(coefs, delays)
    re_max = _envelope_real_axis_crossing(norms, delays)


    mi_vec, abs_omega, gl_s = chain_asymptotes2(coefs, delays)
    print(coefs)
    for g in gl_s:
        print(g[0])


    # for i in range(1, 20):
        
    #     r = _envelope_eval(np.array([re_max - i]), norms, delays)
    #     region = (re_max - i, re_max, -1e-3, r[0])
    #     n = argument_principle_rectangle(lambda x: qpmr.quasipoly.eval(coefs, delays, x), region=region, ds=0.1, eps=1e-6)
    #     print(f"{region}= | {n=}")

    # region_heuristic(coefs, delays, n=30)
        
    # fig, ax = plt.subplots()

    # qpmr.plot.chain_asymptotes(mi_vec, abs_omega, region=region, ax=ax)
    # qpmr.plot.spectrum_envelope(norms, delays, region=region, ax=ax)


    im_end = 300
    n = 100_000

    roots, _ = qpmr_v3(coefs, delays, region)

    qpmr.plot.experimental(roots)
    plt.show()

    for mu, abs_w, gl in zip(mi_vec, abs_omega, gl_s):
        
        gl_coefs, gl_delays = gl
        gl_delays = np.abs(gl_delays - np.max(gl_delays))

        ddelays = delays # - np.max(delays)

        print(gl_delays, ddelays)

        fig, axes = plt.subplots(2, 2)

        # fig, axx = plt.subplots()
        for w in abs_w:
            im_start = w * np.exp(-1/mu*re_max)
            print(re_max, im_start, w)
            # ax.scatter([re_max], [im_start])
            # ax.scatter([0], [w])

            # contour phi
            y = np.linspace(im_start, im_end, n)
            x = -mu * np.log(1./w * y)
            # y = w * np.exp(-1/mu*x)
            # ax.plot(x,y, "-x", alpha=0.2)
            hs = qpmr.quasipoly.eval(coefs, ddelays, x + 1j*y)
            gls = qpmr.quasipoly.eval(gl_coefs, gl_delays, x + 1j*y)

            axes[1][1].plot(y, np.log10(1e-10 + np.abs(hs)), label=f"{mu=}-|w|={w}-g")
            axes[1][1].plot(y, np.log10(1e-10 + np.abs(gls)), label=f"{mu=}-|w|={w}-gl")

            axes[1][0].plot(y, np.log10(1e-10 + np.abs(hs-gls) / np.abs(np.power(x+ 1j * y, 8))), label=f"{mu=}-|w|={w}-|R(s)|")


            axes[0][1].plot( y, np.log10(1e-10 + np.abs(hs-gls) / np.abs(hs)  ), label=f"{mu=}-|w|={w}-rel / hs")
            axes[0][1].plot( y, np.log10(1e-10 + np.abs(hs-gls) / np.abs(gls) ), label=f"{mu=}-|w|={w}-rel / gls")



            # plt.plot(y, np.log10(1e-10 + np.abs(hs - gls)), label=f"{mu=}-|w|={w}")

            # plt.plot(y, np.abs(hs - gls))

            # axx.plot(y, hs.real)
            # axx.plot(y, hs.imag)
            # axx.set_ylim(-1, 100)

        # roots_experimental, _ = qpmr_v3(coefs-gl_coefs, delays, region)

        gl_coefs, gl_delays = qpmr.quasipoly.compress(gl_coefs, gl_delays)
        gl_delays = gl_delays - np.min(gl_delays)
        n0_ix = np.argmax(np.any(gl_coefs!=0.0, axis=0)) # first non-zero index
        
        print(gl_coefs)
        print(f"index={n0_ix}")
        print(gl_coefs[:, n0_ix:], gl_delays, region)
        print(50*"=")
        
        roots_gl, _ = qpmr_v3(gl_coefs[:, n0_ix:], gl_delays, region)

        qpmr.plot.pole_zero(roots, roots_gl, ax=axes[0][0])
        
        # axes[0][0].scatter(roots_experimental.real, roots_experimental.imag, marker="o", linewidths=0.5, color="g", s=15)

        plt.show()
