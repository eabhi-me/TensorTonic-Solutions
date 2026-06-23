import numpy as np

def covariance_matrix(X):
    X = np.asarray(X, dtype=float)

    if X.shape[0] < 2:
        return None

    if X.ndim == 1:
        return None

    mean = np.mean(X, axis=0)
    Xc = X - mean

    n = X.shape[0]

    cov = (Xc.T @ Xc) / (n - 1)

    return cov