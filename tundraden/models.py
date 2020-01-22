import numpy as np
import warnings

from .utils import add_intercept, soft_thresholding
from .cv import leave_one_out_split, kfold_split


class LinearRegression:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef = None
        self.r2 = None

    def fit(self, X, Y):
        if self.fit_intercept:
            X = add_intercept(X)
        self.coef = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
        Y_hat = X.dot(self.coef)
        TSS = np.sum((Y-Y.mean())**2)
        RSS = np.sum((Y-Y_hat)**2)
        self.r2 = 1-RSS/TSS

    def predict(self, X):
        if self.fit_intercept:
            X = add_intercept(X)
        return X.dot(self.coef)


class RidgeRegression:
    def __init__(self, lambda_=1, fit_intercept=True):
        self.lambda_ = lambda_
        self.fit_intercept = fit_intercept
        self.coef = None
        self.test_error = None

    def fit(self, X, Y):
        if self.fit_intercept:
            X = add_intercept(X)
        self.coef = (np.linalg.inv(X.T.dot(X)
                     + self.lambda_*np.identity(X.shape[1])).dot(X.T).dot(Y))

    def predict(self, X):
        if self.fit_intercept:
            X = add_intercept(X)
        return X.dot(self.coef)

    def test(self, X, Y):
        Y_pred = self.predict(X)
        self.test_error = np.mean((Y - Y_pred)**2)
        return self.test_error


class RidgeCV:
    def __init__(self, lambda_grid=None, fit_intercept=True, cv=None):
        if lambda_grid is None:
            lambda_grid = np.linspace(0.1, 1, 10)
        if cv is not None and not isinstance(cv, int):
            raise ValueError(("cv should be either None for leave-one-out or "
                              "an int for k-fold"))
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
        self.coef = (np.linalg.inv(X.T.dot(X)
                     + self.lambda_*np.identity(X.shape[1])).dot(X.T).dot(Y))

    def predict(self, X):
        if self.fit_intercept:
            X = add_intercept(X)
        return X.dot(self.coef)

    def test(self, X, Y):
        Y_pred = self.predict(X)
        self.test_error = np.mean((Y - Y_pred)**2)
        return self.test_error


class NestedRidgeCV:
    def __init__(self, lambda_grid=None, fit_intercept=True, outer_cv=5,
                 inner_cv=10):
        if lambda_grid is None:
            lambda_grid = np.linspace(0.1, 1, 10)
        if inner_cv is not None and not isinstance(inner_cv, int):
            raise ValueError(("inner_cv should be either None for "
                              "leave-one-out CV or an int for k-fold"))
        if outer_cv is not None and not isinstance(outer_cv, int):
            raise ValueError(("outer_cv should be either None for "
                              "leave-one-out CV or an int for k-fold"))
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
        for i, (outer_train_index, outer_test_index) in \
                enumerate(outer_split_function(X)):
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


class LassoRegression:
    def __init__(self, lambda_=1, fit_intercept=True, max_iter=100, tol=0.001,
                 learn_rate=0.01, save_steps=False):
        self.lambda_ = lambda_
        self.fit_intercept = fit_intercept
        self.coef = None
        self.test_error = None
        self.max_iter = max_iter
        self.tol = tol
        self.learn_rate = learn_rate
        self.save_steps = save_steps
        self.steps = []

    def fit(self, X, Y):
        self.steps = []
        if self.fit_intercept:
            X = add_intercept(X)
        self.coef = np.zeros(X.shape[1])
        if self.fit_intercept:
            self.coef[0] = Y.mean()
        if self.save_steps:
            self.steps.append(self.coef)
        learn_rate = self.learn_rate/X.shape[1]
        for i in range(self.max_iter):
            prev_coef = self.coef
            ols_grad = -2*X.T.dot(Y-X.dot(self.coef))
            self.coef = soft_thresholding(self.coef - learn_rate*ols_grad,
                                          self.learn_rate*self.lambda_)
            step = prev_coef-self.coef
            if self.save_steps:
                self.steps.append(self.coef)
            if (i != 0 and
                    np.linalg.norm(step)/np.linalg.norm(prev_coef) < self.tol):
                break
        if i == self.max_iter-1:
            warnings.warn('Max iteration reached!')
        self.steps = np.array(self.steps)

    def predict(self, X):
        if self.fit_intercept:
            X = add_intercept(X)
        return X.dot(self.coef)

    def test(self, X, Y):
        Y_pred = self.predict(X)
        self.test_error = np.mean((Y - Y_pred)**2)
        return self.test_error
