r"""
Package utilities
=================

Logging and other helpers for QPmR.
"""

import logging


def init_qpmr_logger(level=logging.INFO, format: str="%(asctime)s - %(name)s - %(levelname)s - %(message)s") -> logging.Logger:
    """Configure the ``qpmr`` package logger with a stream handler.

    Parameters
    ----------
    level : int, optional
        Logging level (e.g. ``logging.DEBUG``). Default is ``logging.INFO``.
    format : str, optional
        ``logging.Formatter`` format string for console output.

    Returns
    -------
    logger : logging.Logger
        The configured ``qpmr`` logger.

    Notes
    -----
    Each call adds a new ``StreamHandler``. For repeated setup in notebooks or
    scripts, remove existing handlers first if duplicate output is unwanted.
    """
    logger = logging.getLogger("qpmr")
    stream_formatter = logging.Formatter(format)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)
    logger.setLevel(level=level)
    return logger
