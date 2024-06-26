import numpy as np


def generate_data(n: int, m: int, p: int, cov_mat: np.ndarray):
    """generate design matrices and response variables for the labeled and unlabeled data given covariance matrix

    Args:
        n (int): number of labeled data points
        m (int): number of unlabeled data points
        p (int): number of features
        cov_mat (np.ndarray): covariance matrix

    Returns:
        [X_labeled_ols, y_ols]: labeled data for OLS
        [X_labeled_rest, y_rest]: labeled data for the rest of the algorithms
        [X_unlabeled]: unlabeled data
    """
    assert cov_mat.shape == (p, p), "cov_mat must be a square matrix of size p"
    assert np.allclose(cov_mat, cov_mat.T), "cov_mat must be symmetric"
    assert np.all(np.linalg.eigvals(cov_mat) > 0), "cov_mat must be positive definite"

    cov_sqrt = np.linalg.cholesky(cov_mat)

    X_labeled_ols = np.linalg.qr(np.random.randn(n, p))[0] @ cov_sqrt
    X_labeled_rest = np.linalg.qr(np.random.randn(n, p))[0] @ cov_sqrt
    X_unlabeled = np.linalg.qr(np.random.randn(m, p))[0] @ cov_sqrt

    beta = np.random.randn(p)
    assert p >= 3, "p must be greater than or equal to 3"
    # Make first feature irrelevant
    beta[0] = 0
    # The second and third features are correlated with target
    beta[1] = 0.05
    beta[2] = 0.1

    # Fix variance of noise as 1
    y_ols = X_labeled_ols @ beta + np.random.randn(n)
    y_rest = X_labeled_rest @ beta + np.random.randn(n)

    return [X_labeled_ols, y_ols], [X_labeled_rest, y_rest], [X_unlabeled]
