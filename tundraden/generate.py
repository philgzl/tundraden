import numpy as np

from .tsa import generate_arma_samples, generate_varma_samples


def gaussian(mu, covariance_matrix, n_samples):
    X = np.random.multivariate_normal(mu, covariance_matrix, size=n_samples)
    return X


def linear(regressors, weights, sigma):
    noise = np.random.normal(scale=sigma, size=len(regressors))
    return np.dot(regressors, weights) + noise


def arma(ar, ma, n_samples, sigma=1):
    return generate_arma_samples(ar, ma, n_samples, sigma)


def varma(AR, MA, n_samples, Sigma=None):
    return generate_varma_samples(AR, MA, n_samples, Sigma)
