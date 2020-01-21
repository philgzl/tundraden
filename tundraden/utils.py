import numpy as np


def standardize(X):
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    return (X - means)/stds


def covariance_matrix(X):
    return np.cov(X, rowvar=False)


def correlation_matrix(X):
    return np.corrcoef(X, rowvar=False)


def add_intercept(X):
    return np.hstack((np.ones((X.shape[0], 1)), X))


def ReLU(x):
    return x * (x > 0)
