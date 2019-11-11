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


def standarsize(X):
    means = X.mean(axis=0)
    stds = X.mean(axis=0)
    return (X - means)/stds


def covariance_matrix(X):
    return np.cov(X, rowvar=False)


def correlation_matrix(X):
    return np.corrcoef(X, rowvar=False)


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


def train_test_split(X, Y, train_size=None, test_size=None, shuffle=True):
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
    if shuffle:
        indexes = np.random.permutation(n)
    else:
        indexes = np.arange(n)
    X_randomized = X[indexes]
    Y_randomized = Y[indexes]
    X_train, X_test = X_randomized[:train_size], X_randomized[train_size:]
    Y_train, Y_test = Y_randomized[:train_size], Y_randomized[train_size:]
    return X_train, X_test, Y_train, Y_test


def kfold_split(X, n_folds, shuffle=True):
    n = len(X)
    if shuffle:
        indexes = np.random.permutation(n)
    else:
        indexes = np.arange(n)
    limits = np.round(n/n_folds*np.arange(n_folds+1)).astype(int)
    for i in range(n_folds):
        train_index = indexes[np.r_[0:limits[i], limits[i+1]:n]]
        test_index = (indexes[limits[i]:limits[i+1]])
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
    def __init__(self, fit_intercept=False):
        self.fit_intercept = fit_intercept
        self.coef = None

    def fit(self, X, Y):
        if self.fit_intercept:
            X = add_intercept(X)
        self.coef = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

    def predict(self, X):
        if self.fit_intercept:
            X = add_intercept(X)
        return X.dot(self.coef)


class RidgeRegression:
    def __init__(self, lambda_, fit_intercept=False):
        self.lambda_ = lambda_
        self.fit_intercept = fit_intercept
        self.coef = None
        self.test_error = None

    def fit(self, X, Y):
        if self.fit_intercept:
            X = add_intercept(X)
        self.coef = np.linalg.inv(X.T.dot(X) + self.lambda_*np.identity(X.shape[1])).dot(X.T).dot(Y)

    def predict(self, X):
        if self.fit_intercept:
            X = add_intercept(X)
        return X.dot(self.coef)

    def test(self, X, Y):
        Y_pred = self.predict(X)
        self.test_error = np.mean((Y - Y_pred)**2)
        return self.test_error


class RidgeCV:
    def __init__(self, lambda_grid=None, fit_intercept=False, cv=None):
        if lambda_grid is None:
            lambda_grid = np.linspace(0.1, 1, 10)
        if cv is not None and not isinstance(cv, int):
            raise ValueError("'cv' should be either 'None' for leave-one-out CV or an int for k-fold CV")
        self.lambda_grid = lambda_grid
        self.fit_intercept = fit_intercept
        self.cv = cv
        self.coef = None
        self.generalization_errors = None
        self.lambda_ = None
        self.test_error = None

    def fit(self, X, Y):
        if self.cv is None:
            n_folds = len(X)
            split_function = leave_one_out_split
        else:
            n_folds = self.cv
            split_function = lambda X: kfold_split(X, n_folds)
        test_errors = np.zeros((n_folds, len(self.lambda_grid)))
        for i, (train_index, test_index) in enumerate(split_function(X)):
            X_train = X[train_index]
            X_test = X[test_index]
            Y_train = Y[train_index]
            Y_test = Y[test_index]
            for j, lambda_ in enumerate(self.lambda_grid):
                model = RidgeRegression(lambda_)
                model.fit(X_train, Y_train)
                test_errors[i, j] = model.test(X_test, Y_test)
        self.generalization_errors = test_errors.mean(axis=0)
        best_lambda_idx = self.generalization_errors.argmin()
        self.lambda_ = self.lambda_grid[best_lambda_idx]
        if self.fit_intercept:
            X = add_intercept(X)
        self.coef = np.linalg.inv(X.T.dot(X) + self.lambda_*np.identity(X.shape[1])).dot(X.T).dot(Y)

    def predict(self, X):
        if self.fit_intercept:
            X = add_intercept(X)
        return X.dot(self.coef)

    def test(self, X, Y):
        Y_pred = self.predict(X)
        self.test_error = np.mean((Y - Y_pred)**2)
        return self.test_error


class NestedRidgeCV:
    def __init__(self, lambda_grid=None, fit_intercept=False, outer_cv=5, inner_cv=10):
        if lambda_grid is None:
            lambda_grid = np.linspace(0.1, 1, 10)
        if inner_cv is not None and not isinstance(inner_cv, int):
            raise ValueError("'inner_cv' should be either 'None' for leave-one-out CV or an int for k-fold CV")
        if outer_cv is not None and not isinstance(outer_cv, int):
            raise ValueError("'outer_cv' should be either 'None' for leave-one-out CV or an int for k-fold CV")
        self.inner_cv = inner_cv
        self.outer_cv = outer_cv
        self.lambda_grid = lambda_grid
        self.fit_intercept = fit_intercept
        self.models = None

    def run(self, X, Y):
        if self.outer_cv is None:
            n_outer_folds = len(X)
            outer_split_function = leave_one_out_split
        else:
            n_outer_folds = self.outer_cv
            outer_split_function = lambda X: kfold_split(X, n_outer_folds)
        self.models = [RidgeCV(lambda_grid=self.lambda_grid,
                               fit_intercept=self.fit_intercept,
                               cv=self.inner_cv)
                       for i in range(n_outer_folds)]
        for i, (outer_train_index, outer_test_index) in enumerate(outer_split_function(X)):
            X_train = X[outer_train_index]
            X_test = X[outer_test_index]
            Y_train = Y[outer_train_index]
            Y_test = Y[outer_test_index]
            self.models[i].fit(X_train, Y_train)
            self.models[i].test(X_test, Y_test)

    @property
    def test_errors(self):
        return [model.test_error for model in self.models]

    @property
    def generalization_error(self):
        return np.mean(self.test_errors)

    @property
    def lambdas(self):
        return [model.lambda_ for model in self.models]
