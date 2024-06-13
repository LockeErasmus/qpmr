__version__ = "0.0.1.beta0"

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from .qpmr_v2 import qpmr, QpmrOutputMetadata
from .distribution import distribution_diagram