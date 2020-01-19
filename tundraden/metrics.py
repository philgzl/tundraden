import numpy as np

from . import utils
from .models import LinearRegression


def r2(y_true, y_pred):
    '''
    Coefficient of determination.
    '''
    rss = np.sum((y_true - y_pred)**2)
    tss = np.sum((y_true - y_true.mean())**2)
    return 1-rss/tss


def pearson(x, y, standardize=True):
    '''
    Pearson correlation coefficient.

    Parameters:
        x:
            Firt series.
        y:
            Second series.
        standardize:
            Wether to standardize the series before the cross-covariance and
            variances calculation. You should set this to `False` only if you
            know the data comes from a zero-mean and unit-variance
            distribution.

    Returns:
        r_xy:
            The Pearson correlation coefficent between the two series.
    '''
    if standardize:
        x = utils.standardize(x)
        y = utils.standardize(y)
    return np.mean(x*y)


def vif(*args):
    '''
    Variance inflation factor.
    https://en.wikipedia.org/wiki/Variance_inflation_factor

    Two uses cases:
        vif(y_true, y_pred):
            Computes the single VIF factor of the result of a linear regression
            model giving `y_pred` as the estimate of `y_true`. `y_true` and
            `y_pred` must be 1-dimensional and have the same length.
        vif(X):
            Computes a VIF factor for each regressor in `X`, as described in
            https://en.wikipedia.org/wiki/Variance_inflation_factor#Calculation_and_analysis.
            `X` is a 2-dimensional array with size `n_samples*n_regressors`.
    '''
    if len(args) == 2:
        y_true, y_pred = args
        return 1/(1 - r2(y_true, y_pred))
    elif len(args) == 1:
        X = args[0]
        n_vars = X.shape[1]
        vifs = np.zeros(n_vars)
        for i in range(n_vars):
            X_ = np.delete(X, i, 1)
            Y_ = X[:, i]
            coefs = np.linalg.inv(X_.T.dot(X_)).dot(X_.T).dot(Y_)
            Y_pred = X_.dot(coefs)
            vifs[i] = vif(Y_, Y_pred)
        return vifs
    else:
        raise ValueError('vif takes either 1 or 2 arguments')


def partial_r2(X, Y, i=None):
    '''
    Coefficient of partial determination.
    https://en.wikipedia.org/wiki/Coefficient_of_determination#Coefficient_of_partial_determination

    Parameters:
        X:
            Regressors or independent variables, with size
            `n_samples*n_regressors`.
        Y:
            Target or dependent variable, with size `size n_samples*1`.
        i:
            Index of the variable in X for which to compute the coefficient of
            partial determination. If none is provided, then it is calculated
            for every regressor in X.

    Returns:
        r2s:
            Coefficient(s) of partial determination. If `i` is `None`, `r2s` is
            an array containing `n_regressors` values. Otherwise it is just one
            float.
    '''
    if isinstance(i, int):
        lm_full = LinearRegression(fit_intercept=True)
        lm_full.fit(X, Y)
        lm_red = LinearRegression(fit_intercept=True)
        lm_red.fit(np.delete(X, i, 1), Y)
        return (lm_full.r2 - lm_red.r2)#/(1-lm_red.r2)
    elif i is None:
        n_vars = X.shape[1]
        r2s = np.zeros(n_vars)
        for i in range(n_vars):
            r2s[i] = partial_r2(X, Y, i)
        return r2s
    else:
        raise ValueError('i must be either None or an integer')


def _partial_correlation(X, Y, Z, method='recursive'):
    if method == 'recursive':
        if Z.shape[1] == 0:
            return pearson(X, Y)
        else:
            Z0 = Z[:, 0]
            Z = Z[:, 1:]
            r_xy_z = _partial_correlation(X, Y, Z)
            r_xz0_z = _partial_correlation(X, Z0, Z)
            r_yz0_z = _partial_correlation(Z0, Y, Z)
            return (r_xy_z - r_xz0_z*r_yz0_z)/((1-r_xz0_z**2)**0.5 * (1-r_yz0_z**2)**0.5)
    elif method == 'linear_regression':
        lm = LinearRegression(fit_intercept=True)
        lm.fit(Z, X)
        e_x = X - lm.predict(Z)
        lm.fit(Z, Y)
        e_y = Y - lm.predict(Z)
        return pearson(e_x, e_y)
    else:
        raise ValueError('method must be either recursive or linear_regression')


def partial_correlation(X, Y, i=None, method='recursive'):
    if isinstance(i, int):
        Z = np.delete(X, i, 1)
        X = X[:, i]
        return _partial_correlation(X, Y, Z, method)
    elif i is None:
        n_vars = X.shape[1]
        pcorrs = np.zeros(n_vars)
        for i in range(n_vars):
            pcorrs[i] = partial_correlation(X, Y, i, method)
        return pcorrs
    else:
        raise ValueError('i must be either None or an integer')


def _semipartial_correlation(X, Y, Z, method='recursive'):
    if method == 'recursive':
        if Z.shape[1] == 0:
            return pearson(X, Y)
        else:
            Z0 = Z[:, 0]
            Z = Z[:, 1:]
            r_xy_z = _semipartial_correlation(X, Y, Z)
            r_xz0_z = _semipartial_correlation(X, Z0, Z)
            r_yz0_z = _semipartial_correlation(Z0, Y, Z)
            return (r_xy_z - r_xz0_z*r_yz0_z)/((1-r_xz0_z**2)**0.5)
    elif method == 'linear_regression':
        lm = LinearRegression(fit_intercept=True)
        lm.fit(Z, X)
        e_x = X - lm.predict(Z)
        return pearson(e_x, Y)
    else:
        raise ValueError('method must be either recursive or linear_regression')


def semipartial_correlation(X, Y, i=None, method='recursive'):
    if isinstance(i, int):
        Z = np.delete(X, i, 1)
        X = X[:, i]
        return _semipartial_correlation(X, Y, Z, method)
    elif i is None:
        n_vars = X.shape[1]
        spcorrs = np.zeros(n_vars)
        for i in range(n_vars):
            spcorrs[i] = semipartial_correlation(X, Y, i, method)
        return spcorrs
    else:
        raise ValueError('i must be either None or an integer')
