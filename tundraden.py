import numpy as np
from sklearn.linear_model import LinearRegression


def generate_correlated_data(covariance_matrix, n_samples):
    n_regressors = covariance_matrix.shape[0]
    mu = np.zeros(n_regressors)
    X = np.random.multivariate_normal(mu, covariance_matrix, size=n_samples)
    return X


def generate_target(regressors, weights, sigma):
    noise = np.random.normal(scale=sigma, size=len(regressors))
    return np.dot(regressors, weights) + noise


def measure_multicollinearity(X):
    n_vars = X.shape[1]
    vif = np.zeros(n_vars)
    r2 = np.zeros(n_vars)
    model = LinearRegression()
    for i in range(n_vars):
        X_ = np.delete(X, i, 1)
        Y_ = X[:, i]
        model.fit(X_, Y_)
        r2_ = model.score(X_, Y_)
        r2[i] = r2_
        vif[i] = 1/(1-r2_)
    return r2, vif
