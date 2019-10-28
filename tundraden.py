import numpy as np
import matplotlib.pyplot as plt


def generate_correlated_data(covariance_matrix, n_samples):
    n_regressors = covariance_matrix.shape[0]
    mu = np.zeros(n_regressors)
    X = np.random.multivariate_normal(mu, covariance_matrix, size=n_samples)
    return X


def generate_target(regressors, weights, sigma):
    noise = np.random.normal(scale=sigma, size=len(regressors))
    return np.dot(regressors, weights) + noise


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


def add_intercept(X):
    return np.hstack((np.ones((X.shape[0], 1)), X))


def train_test_split(X, Y, train_size=None, test_size=None):
    if train_size is not None and test_size is not None:
        raise ValueError("'train_size' and 'test_size' can't be both specified")
    if len(X) != len(Y):
        raise ValueError("'X' and 'Y' should have the same length")
    n = len(X)
    if train_size is not None:
        if isinstance(train_size, float):
            train_size = round(n*train_size)
    elif test_size is not None:
        if isinstance(test_size, float):
            test_size = round(n*test_size)
        train_size = n - test_size
    else:
        raise ValueError("Either 'train_size' or 'test_size' should be specified")
    randomized_indexes = np.random.permutation(n)
    X_randomized = X[randomized_indexes]
    Y_randomized = Y[randomized_indexes]
    X_train, X_test = X_randomized[:train_size], X_randomized[train_size:]
    Y_train, Y_test = Y_randomized[:train_size], Y_randomized[train_size:]
    return X_train, X_test, Y_train, Y_test


def kfold_split(X, n_folds):
    n = len(X)
    randomized_index = np.random.permutation(n)
    limits = np.round(n/n_folds*np.arange(n_folds+1)).astype(int)
    for i in range(n_folds):
        train_index = randomized_index[np.r_[0:limits[i], limits[i+1]:n]]
        test_index = (randomized_index[limits[i]:limits[i+1]])
        yield train_index, test_index


def leave_one_out_split(X):
    n = len(X)
    index = np.arange(n)
    for i in range(n):
        train_index = np.delete(index, i)
        test_index = np.array([i])
        yield train_index, test_index


def pairplot(X, var_names=None, scatter_kws=None, hist_kws=None, grid_kws=None):
    if scatter_kws is None:
        scatter_kws = {}
    if hist_kws is None:
        hist_kws = {}
    n_vars = X.shape[1]
    for i in range(n_vars):
        for j in range(i+1, n_vars):
            for i_, j_ in [[i, j], [j, i]]:
                plt.subplot(n_vars, n_vars, i_*n_vars+j_+1)
                plt.scatter(X[:, j_], X[:, i_], **scatter_kws)
                if grid_kws is not None:
                    plt.grid(**grid_kws)
                if var_names is not None:
                    if i_ == n_vars-1:
                        plt.xlabel(var_names[j_])
                    if j_ == 0:
                        plt.ylabel(var_names[i_])
    for i in range(n_vars):
        plt.subplot(n_vars, n_vars, i*n_vars+i+1)
        plt.hist(X[:, i], **hist_kws)
        if var_names is not None:
            if i == n_vars-1:
                plt.xlabel(var_names[i])
            if i == 0:
                plt.ylabel(var_names[i])


class LinearRegression:
    def __init__(self, fit_intercept=True):
        self._fit_intercept = fit_intercept
        self.coef = None

    def fit(self, X, Y):
        if self._fit_intercept:
            X = add_intercept(X)
        self.coef = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

    def predict(self, X):
        if self._fit_intercept:
            X = add_intercept(X)
        return X.dot(self.coef)


class RidgeRegression:
    def __init__(self, lambda_, fit_intercept=True):
        self.lambda_ = lambda_
        self._fit_intercept = fit_intercept
        self.coef = None

    def fit(self, X, Y):
        if self._fit_intercept:
            X = add_intercept(X)
        self.coef = np.linalg.inv(X.T.dot(X) + self.lambda_*np.identity(X.shape[1])).dot(X.T).dot(Y)

    def predict(self, X):
        if self._fit_intercept:
            X = add_intercept(X)
        return X.dot(self.coef)
