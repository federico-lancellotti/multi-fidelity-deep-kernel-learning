import numpy as np
from sklearn.neighbors import NearestNeighbors


def kNN(X, n_neighbors, n_jobs):
    neigh = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=n_jobs).fit(X)
    dists, inds = neigh.kneighbors(X)
    return dists, inds


def Levina_Bickel(X, dists, k):
    m = np.log(dists[:, k : k + 1] / dists[:, 1:k])
    m = (k - 2) / np.sum(m, axis=1)
    dim = np.mean(m)
    return dim


def eval_id(X, k_list=20, n_jobs=4):
    if np.isscalar(k_list):
        k_list = np.array([k_list])
    else:
        k_list = np.array(k_list)
    
    kmax = np.max(k_list) + 2
    dists, inds = kNN(X, kmax, n_jobs)
    
    dims = []
    for k in k_list:
        dims.append(Levina_Bickel(X, dists, k))
    
    if len(dims) == 1:
        return dims[0]
    else:
        return np.array(dims)
