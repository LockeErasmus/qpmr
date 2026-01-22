__version__ = "0.0.1"

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from .qpmr_v2 import QpmrInfo
from .qpmr_v3 import qpmr
from .distribution import distribution_diagram, chain_asymptotes, safe_upper_bound_diff
from .utils import init_qpmr_logger
from .utils import init_qpmr_logger as init_logger

from .quasipoly import QuasiPolynomial, TransferFunction
from .quasipoly import examples

