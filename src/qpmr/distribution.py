"""
Set of functions for constructing distribution diagram
"""

import logging

import numpy as np
import numpy.typing as npt

from .quasipoly import QuasiPolynomial

logger = logging.getLogger(__name__)

def concave_envelope_inplace(x, y, mask) -> None:
    """ builds concave envelope out of x, y points
    """
    n = len(mask)
    logger.info(f"{n=}, {x=}, {y=}, {mask=}")
    if n == 0: # empty mask
        return
    elif n == 1:
        mask[0] = True
        return
    elif n == 2:
        mask[0] = True
        mask[1] = True
        return
    else: # at least 3 points
        mask[0] = True
        mask[-1] = True
        x1, y1 = x[0], y[0]
        x2, y2 = x[-1], y[-1]
        distance = ((x2-x1)*(y[1:-1]-y1)-(x[1:-1]-x1)*(y2-y1))
        logger.info(distance)
        i = np.argmax(distance)
        if distance[i] >= 0.0: # extra step is needed, also, i is not index of boundaries
            mask[i+1] = True
            concave_envelope_inplace(x[:i+2], y[:i+2], mask[:i+2])
            concave_envelope_inplace(x[i+1:], y[i+1:], mask[i+1:])

def distribution_diagram(qp: QuasiPolynomial, assume_minimal=False) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """ Creates distribution diagram for quasipolynomial
    
    Args:
        qp (QuasiPolynomial): class representing quasipolynomial
        assume_minimal (bool): wheter to convert qp to minimal form before
            creating distribution diagram, default False
    
    Returns:
        tuple containing

        - thetas (ndarray): max(delays) - delays in ascending order
        - degrees (ndarray): according degree of polynomial
        - mask (ndarray): mask determining concave envelope

    """
    # convert to minimal, sorted form
    if not assume_minimal:
        qp = qp.minimal_form()

    # TODO non-empty
    if qp.empty:
        return
    
    # qp is not empty, at least one delay, degree pair
    delays = qp.delays # positive numbers
    degrees = qp.poly_degrees

    # calculate
    alpha0 = np.max(delays) # 
    thetas = alpha0 - delays # positive numbers, last one should be 0.0

    # concave envelope of x=thetas[::-1], y=degrees[::-1]
    mask = np.full_like(thetas, fill_value=False, dtype=bool)
    mask[0] = True
    mask[-1] = True
    concave_envelope_inplace(thetas[::-1], degrees[::-1], mask)
    
    return thetas[::-1], degrees[::-1], mask


