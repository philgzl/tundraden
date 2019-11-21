import numpy as np


def regressors(covariance_matrix, n_samples):
    n_regressors = covariance_matrix.shape[0]
    mu = np.zeros(n_regressors)
    X = np.random.multivariate_normal(mu, covariance_matrix, size=n_samples)
    return X


def target(regressors, weights, sigma):
    noise = np.random.normal(scale=sigma, size=len(regressors))
    return np.dot(regressors, weights) + noise