try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError("Install matplotlib package via pip to use qpmr.plot")

from .basic import roots, pole_zero, qpmr_contour, argument_principle_circle
from .delay_distribution import spectrum_distribution_diagram, chain_asymptotes
from .complex import experimental