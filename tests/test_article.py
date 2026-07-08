"""
Python tests present in article
-------------------------------

Examples 1 and 2 are MATLAB. Example 4 is too complicated and not necessary to
tests.
"""

def test_example_3(enable_plot: bool):
    """ Example 3: Tests siple quasi-polynomial """
    import qpmr
    import numpy as np

    coefs = np.array([[0, 1],[1, 1],[1, 0.]])
    delays = np.array([0, 1, 2.])

    roots, info = qpmr.qpmr(coefs, delays)
    
    th, deg, m = qpmr.distribution_diagram(
        coefs, delays,
    )

    mu, abs_wk = qpmr.chain_asymptotes(coefs, delays)

    cdp, cdm = qpmr.bounds_neutral_strip(coefs, delays)

    if enable_plot:
        import qpmr.plot
        import matplotlib.pyplot as plt
        
        qpmr.plot.spectrum_distribution_diagram(
            th, deg, m,
        )

        fig, ax = plt.subplots()
        ax.axvline(cdp, color="blue")
        ax.axvline(cdm, color="blue")
        qpmr.plot.chain_asymptotes(
            mu, abs_wk, region=info.region, ax=ax
        )
        qpmr.plot.roots(roots, ax=ax)

        plt.show()

def test_example_5(enable_plot: bool):
    pass





