"""
Functions for determining zero multiplicity
-------------------------------------------

"""

import logging
import numpy as np
import numpy.typing as npt


logger = logging.getLogger(__name__)


def cluster_roots(roots0: npt.NDArray, eps: float) -> tuple[npt.NDArray, npt.NDArray]:
    """ Simple implementation of DBScan with min_neighbors=1

    Approach: cluster -> refine -> count multiplicities (-> verify - this will be done outside of the function)

    Assumptions:
        zeros are separated by more than 4 epsilon

    Args:
        roots0 (array): root candidates

    Notes:
        1. isolation =def= no edges between points in different clusters
        2. eps < ds (grid and eps relationship)
    """

    # step 1: cluster
    n = len(roots0)
    distance_matrix = np.abs( roots0[:,np.newaxis] - roots0[np.newaxis, :] ) # diagonal == 0.0
    adjacency_mask = distance_matrix < eps

    labels = - np.ones(n, dtype=int)
    neighbors = [np.where(adjacency_mask[i])[0] for i in range(n)]
    cluster_id = 0
    for i in range(n):
        print(f"    {i} " )
        if labels[i] != -1:
            continue # already classified into cluster
        
        cluster_id += 1
        labels[i] = cluster_id
        seeds = list(neighbors[i])
        print(neighbors[i])
        seeds.remove(i)
        

        while seeds:
            current = seeds.pop()
            if labels[current] != -1: # already classified into cluster + problem with interconnected clusters
                # WARNING INTERCONNECTED CLUSTERS
                clusters_isolated = False
                continue
            labels[current] = cluster_id
            seeds.extend([n for n in neighbors[current] if labels[n] == -1])
    
    centers = np.zeros(shape=(cluster_id,), dtype=np.complex128)
    multiplicities = np.ones(shape=(cluster_id,), dtype=int)
    for i in range(1, cluster_id+1):
        cluster_mask = labels == i
        centers[i-1] = np.mean( roots0[cluster_mask] )
        multiplicities[i-1] = np.sum(cluster_mask)

    return centers, multiplicities
    
