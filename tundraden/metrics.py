import numpy as np


def r2(y_true, y_pred):
    '''
    Coefficient of determination.
    '''
    total_sq_sum = np.sum((y_true - y_true.mean())**2)
    explained_sq_sum = np.sum((y_pred - y_true.mean())**2)
    return explained_sq_sum/total_sq_sum


def vif(*args):
    '''
    Variance inflation factor.
    https://en.wikipedia.org/wiki/Variance_inflation_factor

    Two uses cases:
    - vif(y_true, y_pred): Computes the single VIF factor of the result of a
                           linear regression model giving `y_pred` as the
                           estimate of `y_true`. `y_true` and `y_pred` must
                           be 1-dimensional and have the same length.
    - vif(X): Computes a VIF factor for each regressor in `X`, as described in
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

    Inputs:
    - `X`: Regressors or independent variables, with size
           `n_samples*n_regressors`.
    - `Y`: Target or dependent variable, with size `size n_samples*1`.
    - `i`: Index of the variable in X for which to compute the coefficient of
           partial determination. If none is provided, then it is calculated
           for every regressor in X.
    Outputs:
    - `r2s`: Coefficient(s) of partial determination. If `i` is `None`, `r2s`
             is an array containing `n_regressors` values. Otherwise it is just
             one float.
    '''
    if isinstance(i, int):
        coefs_full = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
        Y_pred_full = X.dot(coefs_full)
        SS_res_full = np.sum((Y - Y_pred_full)**2)
        X_red = np.delete(X, i, 1)
        coefs_red = np.linalg.inv(X_red.T.dot(X_red)).dot(X_red.T).dot(Y)
        Y_pred_red = X_red.dot(coefs_red)
        SS_res_red = np.sum((Y - Y_pred_red)**2)
        return (SS_res_red - SS_res_full)/SS_res_red
    elif i is None:
        n_vars = X.shape[1]
        r2s = np.zeros(n_vars)
        for i in range(n_vars):
            r2s[i] = partial_r2(X, Y, i)
        return r2s
    else:
        raise ValueError('i must be either None or an integer')
