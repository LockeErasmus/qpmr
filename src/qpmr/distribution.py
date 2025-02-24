"""
Set of functions for constructing distribution diagram
"""

import logging

import numpy as np
import numpy.typing as npt

from .quasipoly import QuasiPolynomial
from .quasipoly.core import poly_degree

logger = logging.getLogger(__name__)

def concave_envelope_inplace(x: npt.NDArray, y: npt.NDArray, mask: npt.NDArray) -> None:
    """ inplace fills the mask with True values representing concave envelope

    Args:
        x (array): x coordinates (thetas) 
        y (array): y coordinates (degrees)
        mask (array): mask 
    """
    n = len(mask)
    logger.info(f"{n=}, {x=}, {y=}, {mask=}")
    if n == 0: # is_empty mask
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

def _distribution_diagram(coefs, delays, **kwargs):

    # TODO has to be in compressed form !!! TODO
    print(f"{delays=}")

    degrees = np.apply_along_axis(poly_degree, 1, coefs)

    # highest degree coefficient - 1D array
    coef_hdeg = coefs[np.arange(0, len(coefs)), degrees.astype(int)]
    thetas = -delays + np.max(delays)

    print(f"{degrees=}")
    print(f"{coef_hdeg=}")
    print(f"{thetas=}")

    # form concave envelope
    # concave envelope of x=thetas[::-1], y=degrees[::-1]
    mask = np.full_like(thetas, fill_value=False, dtype=bool)
    mask[0] = True
    mask[-1] = True
    concave_envelope_inplace(thetas, degrees, mask)
    
    print(f"{mask=}")
    print(f"{degrees[mask]=}")
    print(f"{thetas[mask]=}")
    
    ix = np.arange(0, len(mask))[mask]
    n_segments = np.sum(mask) - 1
    logger.debug(f"Number of segments: {n_segments}") # TODO what if one delay only
    for k in range(n_segments): # iterate over all segments
        logger.debug(f"Working on segment {k} -> spaning {ix[k]} - {ix[k+1]}")

        # TODO: vectorize
        segment_mi = (degrees[ix[k+1]] - degrees[ix[k]]) / (thetas[ix[k+1]] - thetas[ix[k]]) # slope
        print(f"{segment_mi=}")

        if segment_mi == 0.0: # neutral segment
            # TODO
            pass
        else:
            # construct coefficients of polynomial fr(w)
            m0 = degrees[ix[k]]
            deg_ = degrees[ix[k]:ix[k+1]+1] - m0
            coef_ = coef_hdeg[ix[k]:ix[k+1]+1]

            print(deg_, coef_)

            segment_coef = np.zeros(shape=(np.max(deg_)+1))
            segment_coef[0] = coef_hdeg[ix[k]]
            for i in range(1, np.max(deg_) + 1):
                print(i)
                segment_coef[i] = np.sum(coef_[deg_ == (i - 1)])

        logger.debug(f"{segment_coef=}")

        raise NotImplementedError(".")

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
        qp = qp.compressed

    # TODO non-is_empty
    if qp.is_empty:
        return
    
    # qp is not is_empty, at least one delay, degree pair
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


