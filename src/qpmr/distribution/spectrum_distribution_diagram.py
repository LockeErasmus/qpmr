"""
Spectrum distribution diagram
-----------------------------
"""
from dataclasses import dataclass
import logging

import numpy as np
import numpy.typing as npt

from qpmr.quasipoly.core import poly_degree, compress

logger = logging.getLogger(__name__)

def _concave_envelope_inplace(x: npt.NDArray, y: npt.NDArray, mask: npt.NDArray) -> None:
    """Fill ``mask`` in place with the concave envelope of ``(x, y)``."""
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
    """Return a boolean mask for the concave envelope of diagram points.

    Parameters
    ----------
    x : ndarray
        Theta coordinates (ascending).
    y : ndarray
        Polynomial degrees.

    Returns
    -------
    mask : ndarray
        ``True`` for envelope vertices.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` are not 1D arrays of equal length.
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

@dataclass
class SpectrumDistributionDiagramMetadata:
    """Cached spectrum distribution diagram data.

    Attributes
    ----------
    P_theta : ndarray or None
        Theta coordinates of diagram points.
    P_degree : ndarray or None
        Polynomial degrees at each point.
    mask : ndarray or None
        Boolean mask selecting the concave envelope.
    """
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
    """Build the spectrum distribution diagram for a quasi-polynomial.

    Parameters
    ----------
    coefs : ndarray
        Matrix of polynomial coefficients. Each row represents the coefficients
        corresponding to a specific delay.
    delays : ndarray
        Vector of delays associated with each row in ``coefs``.
    assume_compressed : bool, optional
        If ``False`` (default), compress and sort delays before analysis.

    Returns
    -------
    thetas : ndarray
        ``max(delays) - delays`` in ascending order.
    degrees : ndarray
        Degree of the polynomial factor at each diagram point.
    mask : ndarray
        Boolean mask selecting the concave envelope vertices.
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

