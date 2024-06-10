import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


def kNN(X, n_neighbors, n_jobs):
    """
    Computes the k-nearest neighbors of each point in the input data.

    Parameters:
    X (array-like): The input data.
    n_neighbors (int): The number of neighbors to consider.
    n_jobs (int): The number of parallel jobs to run for neighbor search.

    Returns:
    dists (array-like): The distances to the k-nearest neighbors for each point.
    inds (array-like): The indices of the k-nearest neighbors for each point.
    """

    neigh = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=n_jobs).fit(X)
    dists, inds = neigh.kneighbors(X)
    return dists, inds


def Levina_Bickel(dists, k):
    """
    Calculate the intrinsic dimensionality of a dataset using the Levina-Bickel estimator.

    Parameters:
    dists (array-like): Pairwise distances between points in the dataset.
    k (int): The number of nearest neighbors to consider.

    Returns:
    dim (float): The estimated intrinsic dimensionality of the dataset.
    """
    
    epsilon = 1e-10 # small value to avoid division by zero
    dists = np.where(dists == 0, epsilon, dists)

    m = np.log(dists[:, k:k+1] / dists[:, 1:k])
    m = (k - 2) / np.sum(m, axis=1)
    dim = np.mean(m)
    return dim


def eval_id(X, k_list=20, n_jobs=4):
    """
    Evaluate the intrinsic dimensionality of a dataset using the Levina-Bickel method.

    Parameters:
        - X: numpy array
            The input dataset.
        - k_list: int or list, optional (default=20)
            The number of nearest neighbors to consider for each intrinsic dimension estimation.
            If an integer is provided, the same number of neighbors will be used for all estimations.
            If a list is provided, a different number of neighbors can be specified for each estimation.
        - n_jobs: int, optional (default=4)
            The number of parallel jobs to run for computing the nearest neighbors.

    Returns:
        - dims: float or numpy array
            The estimated intrinsic dimensions of the dataset.
            If a single value of k_list is provided, a float is returned.
            If multiple values of k_list are provided, a numpy array is returned.
    """

    # Convert input to numpy array if it is a PyTorch tensor
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()

    if np.isscalar(k_list):
        k_list = np.array([k_list])
    else:
        k_list = np.array(k_list)
    
    kmax = np.max(k_list) + 2
    dists, inds = kNN(X, kmax, n_jobs)
    
    dims = []
    for k in k_list:
        dims.append(Levina_Bickel(dists, k))
    
    if len(dims) == 1:
        return dims[0]
    else:
        return np.array(dims)

 
def estimate_ID(z_LF, z_next_LF, z_fwd_LF):
    """
    Estimates the intrinsic dimension (ID) based on the given low-fidelity (LF) latent representation
    of the system.

    Parameters:
    z_LF (array-like): Low-fidelity latent variables at the current time instant.
    z_next_LF (array-like): Low-fidelity latent variables at the next time instant.
    z_fwd_LF (array-like): Low-fidelity latent variables produced by the forward part of the model.

    Returns:
    int: The estimated intrinsic dimension (ID) based on the given low-fidelity latent variables.
    """

    ID_0 = eval_id(z_LF)
    ID_1 = eval_id(z_next_LF)
    ID_fwd = eval_id(z_fwd_LF)
    
    print("ID_0=", ID_0, ", ID_1=", ID_1, ", ID_fwd=", ID_fwd)
    
    ID = int(round((ID_0 + ID_1 + ID_fwd)/3))

    return ID
