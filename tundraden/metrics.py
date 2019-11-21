import numpy as np

from .models import LinearRegression


def r2(y_true, y_pred):
    total_sq_sum = np.sum((y_true - y_true.mean())**2)
    explained_sq_sum = np.sum((y_pred - y_true.mean())**2)
    return explained_sq_sum/total_sq_sum


def vif(y_true, y_pred):
    return 1/(1 - r2(y_true, y_pred))


def partial_r2(X):
    n_vars = X.shape[1]
    r2s = np.zeros(n_vars)
    model = LinearRegression()
    for i in range(n_vars):
        X_ = np.delete(X, i, 1)
        Y_ = X[:, i]
        model.fit(X_, Y_)
        Y_pred = model.predict(X_)
        r2s[i] = r2(Y_, Y_pred)
    return r2s


def partial_vif(X):
    n_vars = X.shape[1]
    vifs = np.zeros(n_vars)
    model = LinearRegression()
    for i in range(n_vars):
        X_ = np.delete(X, i, 1)
        Y_ = X[:, i]
        model.fit(X_, Y_)
        Y_pred = model.predict(X_)
        vifs[i] = vif(Y_, Y_pred)
    return vifs
