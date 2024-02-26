import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from .qpmr_v2 import qpmr, QpmrOutputMetadata