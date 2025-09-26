"""
Set of tests for novel region heuristic
---------------------------------------

"""
import pytest

import qpmr.quasipoly.examples as examples
from qpmr.region_heuristic import region_heuristic
from qpmr.qpmr_v3 import qpmr as qpmr_v3
from qpmr.qpmr_v3 import QpmrSubInfo, QpmrRecursionContext
from qpmr.distribution.zero_chains import chain_asymptotes, chain_asymptotes2
from qpmr.argument_principle import argument_principle_rectangle
import qpmr.quasipoly

from qpmr.distribution.envelope_curve import _spectral_norms, _envelope_real_axis_crossing, _envelope_eval
import qpmr.plot

@pytest.mark.parametrize(
    argnames="qp, qpmr_args, qpmr_kwargs",
    argvalues=[
        (examples.vyhlidal2014qpmr_02(), (), {}),
    ],
    ids=[
        "vyhlidal2014qpmr-02",
    ],
)
def test_region_heuristic(qp, qpmr_args: tuple, qpmr_kwargs: dict, enable_plot: bool):

    import numpy as np
    
    coefs, delays = qp
    region=(-10, 20, 0, 100)

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

    if enable_plot:
        import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()

        # qpmr.plot.chain_asymptotes(mi_vec, abs_omega, region=region, ax=ax)
        # qpmr.plot.spectrum_envelope(norms, delays, region=region, ax=ax)


        im_end = 300
        n = 100_000

        for mu, abs_w, gl in zip(mi_vec, abs_omega, gl_s):
            
            gl_coefs, gl_delays = gl

            ddelays = delays # - np.max(delays)

            print(gl_delays, ddelays)

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

                

                plt.plot(y, np.log10(1e-10 + np.abs(hs)), label=f"{mu=}-|w|={w}-g")
                plt.plot(y, np.log10(1e-10 + np.abs(gls)), label=f"{mu=}-|w|={w}-gl")

                # plt.plot(y, np.log10(1e-10 + np.abs(hs - gls)), label=f"{mu=}-|w|={w}")

                # plt.plot(y, np.abs(hs - gls))

                # axx.plot(y, hs.real)
                # axx.plot(y, hs.imag)
                # axx.set_ylim(-1, 100)

            

                
        # qpmr.plot.qpmr_solution_tree(ctx, ax=ax)
        plt.legend()
        plt.show()
