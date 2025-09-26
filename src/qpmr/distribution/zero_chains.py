"""
Algorithm for detecting chains of zeros
"""
import logging

import numpy as np
import numpy.typing as npt

from qpmr.quasipoly.core import poly_degree, compress
from .spectrum_distribution_diagram import distribution_diagram

logger = logging.getLogger(__name__)

def chain_asymptotes(coefs: npt.NDArray, delays: npt.NDArray, **kwargs):
    """ Calculates exponentials of the root chains

    Args:
        coefs (array): matrix definition of polynomial coefficients (each row
            represents polynomial coefficients corresponding to delay)
        delays (array): vector definition of associated delays (each delay
            corresponds to row in `coefs`)
        **kwargs:
            assume_compressed (bool): set to True if quasipolynomial already
                in compressed form, default False
            abs_wk_decimals_round (int): absolute value of wk is rounded to this
                amount of decimals before selecting unique values, default 10
    
    Returns:
        tuple containing:

            - mi_vec (array): segment slopes
            - abs_omega (list of `array`): corresponding unique absolute values
                of roots    
    """
    abs_wk_decimals_round = kwargs.get("abs_wk_decimals_round", 10)
    assume_compressed = kwargs.get("assume_compressed", False)
    if assume_compressed:
        logger.debug("Not checking if input quasipolynomial is compressed")
    else: # perform compression and delay sorting
        coefs, delays = compress(coefs, delays)

    # step 1 - obtain `spectrum distribution diagram` (SDD)
    thetas, degrees, mask = distribution_diagram(coefs, delays, assume_compressed=True)

    # TODO - degenerate cases of 1 delay ...

    # step 2 - for each segment of SDD, obtain mi_k and set of unique |w_k|,
    # where w_k is root of polynomial fr(w)

    # highest degree coefficient corresponding to points in SDD
    coef_hdeg = (coefs[np.arange(len(coefs)-1,-1,-1), degrees.astype(int)])
    
    ix = np.arange(0, len(mask))[mask]
    n_segments = np.sum(mask) - 1
    mi_vec = np.zeros(shape=(n_segments,), dtype=np.float64) # storing mi
    poly_fw = [] # container for associated polynomial representation
    roots = [] # container for roots
    abs_omega = []

    logger.debug(f"Number of segments: {n_segments}") # TODO what if one delay only
    for k in range(n_segments): # iterate over all segments
        point_start = (thetas[ix[k]], degrees[ix[k]])
        point_end = (thetas[ix[k+1]], degrees[ix[k+1]])
        logger.debug(f"Working on segment L_{k} = [P_{ix[k]}={point_start}, ... , P_{ix[k+1]}]={point_end}")

        # TODO: vectorize
        segment_mi = (degrees[ix[k+1]] - degrees[ix[k]]) / (thetas[ix[k+1]] - thetas[ix[k]]) # slope
        mi_vec[k] = segment_mi

        if segment_mi == 0.0: # this is neutral segment
            logger.debug(f"Neutral segment")
            # TODO
            poly_fw.append(np.array([], dtype=np.float64))
            roots.append(np.array([], dtype=np.float64))
            abs_omega.append(np.array([], dtype=np.float64))
        elif segment_mi < 0: # advanced segment
            m_end = degrees[ix[k+1]] # degree of P_{k+1} (right, ending point)
            segment_rel_deg = degrees[ix[k]:ix[k+1]+1] - m_end
            segment_coef_all = coef_hdeg[ix[k]:ix[k+1]+1]

            logger.debug(f"relative degrees: {segment_rel_deg}")

            segment_coef = np.zeros(shape=(np.abs(np.max(segment_rel_deg)) + 1))
            # pick correct coefficients and form polynomial fr(w) representation
            prev_d = -1 # first d=0
            for d, c in zip(segment_rel_deg[::-1], segment_coef_all[::-1]):
                if d > prev_d:
                    # adding to representation
                    segment_coef[d] = c
                    prev_d = d
            
            # TODO minimal form of polynomial ? or unnecessary?
            segment_roots = np.roots(segment_coef[::-1]) # TODO
            wk = np.round(np.abs(segment_roots), decimals=abs_wk_decimals_round)
            wk = np.unique(wk)
            logger.debug(f"{segment_coef=}, {segment_roots=}, |wk|={wk}")
            poly_fw.append(segment_coef)
            roots.append(segment_roots)

            # we need only the roots with unique absolute value
            abs_omega.append(wk)


        else: # retarded segment
            # construct coefficients of polynomial fr(w)
            m0 = degrees[ix[k]] # degree of left P
            # all Points P with relative degrees and coefficients associated
            # with the highest power s
            segment_rel_deg = degrees[ix[k]:ix[k+1]+1] - m0
            segment_coef_all = coef_hdeg[ix[k]:ix[k+1]+1]

            logger.debug(f"relative degrees: {segment_rel_deg}")

            segment_coef = np.zeros(shape=(np.abs(np.max(segment_rel_deg[1:])) + 1))
            # pick correct coefficients and form polynomial fr(w) representation
            prev_d = -1 # first d=0
            for d, c in zip(segment_rel_deg, segment_coef_all):
                if d > prev_d:
                    # adding to representation
                    segment_coef[d] = c
                    prev_d = d
            
            # TODO minimal form of polynomial ? or unnecessary?
            segment_roots = np.roots(segment_coef[::-1])
            wk = np.round(np.abs(segment_roots), decimals=abs_wk_decimals_round)
            wk = np.unique(wk)
            logger.debug(f"{segment_coef=}, {segment_roots=}, |wk|={wk}")
            poly_fw.append(segment_coef)
            roots.append(segment_roots)

            # we need only the roots with unique absolute value
            abs_omega.append(wk)
    
    return mi_vec, abs_omega


def chain_asymptotes2(coefs: npt.NDArray, delays: npt.NDArray, **kwargs):
    """ Calculates exponentials of the root chains

    Args:
        coefs (array): matrix definition of polynomial coefficients (each row
            represents polynomial coefficients corresponding to delay)
        delays (array): vector definition of associated delays (each delay
            corresponds to row in `coefs`)
        **kwargs:
            assume_compressed (bool): set to True if quasipolynomial already
                in compressed form, default False
            abs_wk_decimals_round (int): absolute value of wk is rounded to this
                amount of decimals before selecting unique values, default 10
    
    Returns:
        tuple containing:

            - mi_vec (array): segment slopes
            - abs_omega (list of `array`): corresponding unique absolute values
                of roots    
    """
    abs_wk_decimals_round = kwargs.get("abs_wk_decimals_round", 10)
    assume_compressed = kwargs.get("assume_compressed", False)
    if assume_compressed:
        logger.debug("Not checking if input quasipolynomial is compressed")
    else: # perform compression and delay sorting
        coefs, delays = compress(coefs, delays)

    # step 1 - obtain `spectrum distribution diagram` (SDD)
    thetas, degrees, mask = distribution_diagram(coefs, delays, assume_compressed=True)

    # TODO - degenerate cases of 1 delay ...

    # step 2 - for each segment of SDD, obtain mi_k and set of unique |w_k|,
    # where w_k is root of polynomial fr(w)

    # highest degree coefficient corresponding to points in SDD
    coef_hdeg = (coefs[np.arange(len(coefs)-1,-1,-1), degrees.astype(int)])
    
    ix = np.arange(0, len(mask))[mask]
    n_segments = np.sum(mask) - 1
    mi_vec = np.zeros(shape=(n_segments,), dtype=np.float64) # storing mi
    poly_fw = [] # container for associated polynomial representation
    roots = [] # container for roots
    abs_omega = []
    gl_s = []

    logger.debug(f"Number of segments: {n_segments}") # TODO what if one delay only
    for k in range(n_segments): # iterate over all segments
        point_start = (thetas[ix[k]], degrees[ix[k]])
        point_end = (thetas[ix[k+1]], degrees[ix[k+1]])
        logger.debug(f"Working on segment L_{k} = [P_{ix[k]}={point_start}, ... , P_{ix[k+1]}]={point_end}")

        # TODO: vectorize
        segment_mi = (degrees[ix[k+1]] - degrees[ix[k]]) / (thetas[ix[k+1]] - thetas[ix[k]]) # slope
        mi_vec[k] = segment_mi

        if segment_mi == 0.0: # this is neutral segment
            logger.debug(f"Neutral segment")
            # TODO
            poly_fw.append(np.array([], dtype=np.float64))
            roots.append(np.array([], dtype=np.float64))
            abs_omega.append(np.array([], dtype=np.float64))
        elif segment_mi < 0: # advanced segment
            m_end = degrees[ix[k+1]] # degree of P_{k+1} (right, ending point)
            segment_rel_deg = degrees[ix[k]:ix[k+1]+1] - m_end
            segment_coef_all = coef_hdeg[ix[k]:ix[k+1]+1]

            logger.debug(f"relative degrees: {segment_rel_deg}")

            segment_coef = np.zeros(shape=(np.abs(np.max(segment_rel_deg)) + 1))
            # pick correct coefficients and form polynomial fr(w) representation
            prev_d = -1 # first d=0
            for d, c in zip(segment_rel_deg[::-1], segment_coef_all[::-1]):
                if d > prev_d:
                    # adding to representation
                    segment_coef[d] = c
                    prev_d = d
            
            # TODO minimal form of polynomial ? or unnecessary?
            segment_roots = np.roots(segment_coef[::-1]) # TODO
            wk = np.round(np.abs(segment_roots), decimals=abs_wk_decimals_round)
            wk = np.unique(wk)
            logger.debug(f"{segment_coef=}, {segment_roots=}, |wk|={wk}")
            poly_fw.append(segment_coef)
            roots.append(segment_roots)

            # we need only the roots with unique absolute value
            abs_omega.append(wk)


        else: # retarded segment
            # construct coefficients of polynomial fr(w)
            m0 = degrees[ix[k]] # degree of left P
            # all Points P with relative degrees and coefficients associated
            # with the highest power s
            indexes = [i for i in range(ix[k], ix[k+1]+1)]
            segment_rel_deg = degrees[ix[k]:ix[k+1]+1] - m0
            segment_coef_all = coef_hdeg[ix[k]:ix[k+1]+1]


            logger.debug(f"relative degrees: {segment_rel_deg}")
            
            gl_coefs = np.zeros_like(coefs)
            segment_coef = np.zeros(shape=(np.abs(np.max(segment_rel_deg[1:])) + 1))
            # pick correct coefficients and form polynomial fr(w) representation
            prev_d = -1 # first d=0
            for i, d, c in zip(indexes, segment_rel_deg, segment_coef_all):
                if d > prev_d:
                    # adding to representation
                    segment_coef[d] = c
                    prev_d = d

                    gl_coefs[i, d + m0] = c
            
            # TODO minimal form of polynomial ? or unnecessary?
            segment_roots = np.roots(segment_coef[::-1])
            
            wk = np.round(np.abs(segment_roots), decimals=abs_wk_decimals_round)
            wk = np.unique(wk)
            logger.debug(f"{segment_coef=}, {segment_roots=}, |wk|={wk}")
            poly_fw.append(segment_coef)
            roots.append(segment_roots)

            # we need only the roots with unique absolute value
            abs_omega.append(wk)

            # g_l(s) quasi-polynomials
            gl_s.append(
                (gl_coefs, np.copy(thetas))
            )

    
    return mi_vec, abs_omega, gl_s