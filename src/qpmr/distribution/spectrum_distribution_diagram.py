"""
Spectrum distribution diagram
-----------------------------
"""
import logging

import numpy as np
import numpy.typing as npt

from qpmr.quasipoly.core import poly_degree, compress

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
        x (array): x coordinates (thetas) in ascending order 
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


class SpectrumDistributionDiagramMetadata:
    # TODO consider moving to dataclass implementation
    P_theta: npt.NDArray = None
    P_degree: npt.NDArray = None
    mask: npt.NDArray = None

    @property
    def L_theta(self) -> npt.NDArray:
        if self.P_theta:
            return self.P_theta[self.mask]
        else:
            return None
    
    @property
    def L_degree(self) -> npt.NDArray:
        if self.P_degree:
            return self.P_degree[self.mask]
        else:
            return None

def distribution_diagram(coefs, delays, **kwargs) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """ Creates spectrum distribution diagram for quasipolynomial
    
    Args:
        TODO
        qp (QuasiPolynomial): class representing quasipolynomial

        assume_minimal (bool): wheter to convert qp to minimal form before
            creating distribution diagram, default False
    
    Returns:
        tuple containing

        - thetas (ndarray): max(delays) - delays in ascending order
        - degrees (ndarray): according degree of polynomial
        - mask (ndarray): mask determining concave envelope
    """
    assume_compressed = kwargs.get("assume_compressed", False)
    if assume_compressed:
        logger.debug("Not checking if input quasipolynomial is compressed")
    else: # perform compression and delay sorting
        coefs, delays = compress(coefs, delays)

    # TODO empty quasipolynomial and/or 1 delay

    # 1D vector representing the degree of polynomials
    degrees = np.apply_along_axis(poly_degree, 1, coefs[::-1, :])
    # highest degree coefficient and thetas
    coef_hdeg = (coefs[np.arange(len(coefs)-1,-1,-1), degrees.astype(int)])
    thetas = -delays[::-1] + np.max(delays)

    # form concave envelope (represented by mask), i.e.
    # envelope [x,y] is formed via thetas[mask], degrees[mask] 
    mask = _concave_envelope(thetas, degrees)

    return thetas, degrees, mask

