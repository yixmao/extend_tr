import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform

def tversky_distance(fp1, fp2, alpha, beta):
    """
    Computes the Tversky index between two binary fingerprints.

    Parameters:
    - fp1: array-like, first binary fingerprint
    - fp2: array-like, second binary fingerprint
    - alpha: float, weight for the penalty for elements in fp1 but not in fp2
    - beta: float, weight for the penalty for elements in fp2 but not in fp1

    Returns:
    - score: float, the Tversky index (ranges from 0 to 1)
    """
    fp1 = np.asarray(fp1, dtype=bool)
    fp2 = np.asarray(fp2, dtype=bool)
    intersection = np.sum(fp1 & fp2)
    diff_fp1_fp2 = np.sum(fp1 & ~fp2)
    diff_fp2_fp1 = np.sum(fp2 & ~fp1)
    # calculate tversky distance
    score = intersection / (intersection + alpha * diff_fp1_fp2 + beta * diff_fp2_fp1)
    distance  = 1-score
    return distance


def calculate_distances(data, args):
    """
    Calculates the pairwise distance matrix based on the specified metric.

    Parameters:
    - data: array-like, the descriptors for which distances will be calculated
    - args: Namespace, contains the selected distance metric and additional parameters:
        - args.distance: str, the distance metric (e.g., 'jaccard', 'euclidean', 'tversky')
        - args.tversky_alpha: float, alpha parameter for Tversky distance (if applicable)
        - args.tversky_beta: float, beta parameter for Tversky distance (if applicable)

    Returns:
    - distance_matrix: np.ndarray, matrix of pairwise distances
    """
    if args.distance in ['jaccard', 'euclidean', 'manhattan', 'cosine', 'chebyshev']:
        return pairwise_distances(data, metric=args.distance, n_jobs=-1)
    elif args.distance == 'tversky':
        return squareform(pdist(data, lambda u, v: tversky_distance(u, v, args.tversky_alpha, args.tversky_beta)))
    else:
        raise ValueError("Unsupported metric")
