"""
Functions for determining zero multiplicity
-------------------------------------------

"""

import logging
import numpy as np
import numpy.typing as npt


logger = logging.getLogger(__name__)


def cluster_roots(roots0: npt.NDArray, eps: float) -> tuple[npt.NDArray, npt.NDArray]:
    """Cluster nearby root candidates and estimate multiplicities.

    Simple DBSCAN-like clustering with ``min_neighbors=1``. Assumes distinct
    zeros are separated by more than ``4 * eps``.

    Parameters
    ----------
    roots0 : ndarray
        1D array of complex root candidates from spectrum mapping.
    eps : float
        Maximum distance for two candidates to belong to the same cluster.
        Should satisfy ``eps < ds`` where ``ds`` is the mapping grid step.

    Returns
    -------
    centers : ndarray
        Cluster centroids (refined root approximations).
    multiplicities : ndarray of int
        Number of candidates in each cluster.

    Notes
    -----
    Isolated clusters have no edges between points in different components.
    Full multiplicity verification is performed outside this function.
    """

    # step 1: cluster
    n = len(roots0)
    distance_matrix = np.abs( roots0[:,np.newaxis] - roots0[np.newaxis, :] )
    adjacency_mask = distance_matrix < eps

    labels = - np.ones(n, dtype=int)
    neighbors = [np.where(adjacency_mask[i])[0] for i in range(n)]
    cluster_id = 0
    for i in range(n):
        if labels[i] != -1:
            continue

        cluster_id += 1
        labels[i] = cluster_id
        seeds = list(neighbors[i])
        seeds.remove(i)

        while seeds:
            current = seeds.pop()
            if labels[current] != -1:
                clusters_isolated = False
                continue
            labels[current] = cluster_id
            seeds.extend([n for n in neighbors[current] if labels[n] == -1])

    centers = np.zeros(shape=(cluster_id,), dtype=np.complex128)
    multiplicities = np.ones(shape=(cluster_id,), dtype=int)
    for i in range(1, cluster_id+1):
        cluster_mask = labels == i
        centers[i-1] = np.mean( roots0[cluster_mask] )
        logger.debug(roots0[cluster_mask])
        multiplicities[i-1] = np.sum(cluster_mask)

    return centers, multiplicities
