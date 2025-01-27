"""
Set of utility functions
"""

import logging

def init_qpmr_logger(level=logging.INFO, format: str="%(asctime)s - %(name)s - %(levelname)s - %(message)s") -> logging.Logger:
    """ Initializes QPmR logger with Streamhandler and level """
    logger = logging.getLogger("qpmr")
    stream_formatter = logging.Formatter(format)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)
    logger.setLevel(level=level)
    return logger
