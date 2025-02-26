"""
Set of functions for examining distribution of zeros
----------------------------------------------------
1. distribution diagram
2. zeros-chain "asymptotes"
3. areas free of zeros - TODO
"""

import logging

import numpy as np
import numpy.typing as npt

from .quasipoly import QuasiPolynomial
from .quasipoly.core import poly_degree, compress

logger = logging.getLogger(__name__)

def _concave_envelope_inplace(x: npt.NDArray, y: npt.NDArray, mask: npt.NDArray) -> None:
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
            _concave_envelope_inplace(x[:i+2], y[:i+2], mask[:i+2])
            _concave_envelope_inplace(x[i+1:], y[i+1:], mask[i+1:])

def _concave_envelope(x: npt.NDArray, y: npt.NDArray) -> npt.NDArray:
    """ Creates concave envelope mask

    Args:
        x (array): x coordinates (thetas) 
        y (array): y coordinates (degrees)
    
    Returns:
        mask (array): mask defining which [x,y] forms the envelope
    """
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs have to 1D arrays")
    if x.shape != y.shape:
        raise ValueError(f"Length of `x` and `y` has to match")
    if x.size == 0: # Return None if empty
        return 
    
    mask = np.full_like(x, fill_value=False, dtype=bool)
    mask[0] = True
    mask[-1] = True
    _concave_envelope_inplace(x,y,mask)
    return mask


def _distribution_diagram(coefs, delays, **kwargs):
    """ TODO
    
    Args:
        coefs (array): matrix definition of polynomial coefficients (each row
            represents polynomial coefficients corresponding to delay)
        delays (array): vector definition of associated delays (each delay
            corresponds to row in `coefs`)
        **kwargs:
            assume_compressed (bool): set to True if quasipolynomial already
                in compressed form, default False
    Returns:
        TODO
    """
    assume_compressed = kwargs.get("assume_compressed", False)
    if assume_compressed:
        logger.debug("Not checking if input quasipolynomial is ")
    else: # perform compression and delay sorting
        coefs, delays = compress(coefs, delays)
    
    # 1D vector representing the degree of polynomials
    degrees = np.apply_along_axis(poly_degree, 1, coefs[::-1, :])
    # highest degree coefficient and thetas
    coef_hdeg = (coefs[np.arange(len(coefs)-1,-1,-1), degrees.astype(int)])
    thetas = -delays[::-1] + np.max(delays)

    # form concave envelope (represented by mask), i.e.
    # envelope [x,y] is formed via thetas[mask], degrees[mask] 
    mask = _concave_envelope(thetas, degrees)
    
    print(f"{mask=}")
    print(f"{degrees[mask]=}")
    print(f"{thetas[mask]=}")
    
    ix = np.arange(0, len(mask))[mask]
    n_segments = np.sum(mask) - 1
    mi_vec = np.zeros(shape=(n_segments,), dtype=np.float64) # storing mi
    poly_fw = [] # container for associated polynomial representation
    roots = [] # container for roots
    abs_omega = []

    logger.debug(f"Number of segments: {n_segments}") # TODO what if one delay only
    for k in range(n_segments): # iterate over all segments
        logger.debug(f"Working on segment {k} -> spaning {ix[k]} - {ix[k+1]}")

        # TODO: vectorize
        segment_mi = (degrees[ix[k+1]] - degrees[ix[k]]) / (thetas[ix[k+1]] - thetas[ix[k]]) # slope
        mi_vec[k] = segment_mi

        if segment_mi == 0.0: # this is neutral segment
            logger.debug(f"Neutral segment")
            # TODO
        else:
            # construct coefficients of polynomial fr(w)
            m0 = degrees[ix[k]] # degree of left P
            # all Points P with relative degrees and coefficients associated
            # with the highest power s
            segment_rel_deg = degrees[ix[k]:ix[k+1]+1] - m0
            segment_coef_all = coef_hdeg[ix[k]:ix[k+1]+1]

            segment_coef = np.zeros(shape=(np.max(segment_rel_deg)+1))
            # pick correct coefficients and form polynomial fr(w) representation
            prev_d = -1 # first d=0
            for d,c in zip(segment_rel_deg, segment_coef_all):
                if d > prev_d:
                    # adding to representation
                    segment_coef[d] = c
                    prev_d = d
            # TODO minimal form of polynomial ? or unnecessary?
            segment_roots = np.roots(segment_coef[::-1])
            wk = np.round(np.abs(segment_roots), decimals=10)
            wk = np.unique(wk)
            logger.debug(f"{segment_coef=}, {segment_roots=}, |wk|={wk}")
            poly_fw.append(segment_coef)
            roots.append(segment_roots)

            # we need only the roots with unique absolute value
            abs_omega.append(wk)
    
    return mi_vec, abs_omega


def _chain_asymptotes():
    """
    
    """



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
    mask = _concave_envelope(thetas[::-1], degrees[::-1], mask)
    
    return thetas[::-1], degrees[::-1], mask


