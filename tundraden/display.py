import matplotlib.pyplot as plt

from . import tsa


def pairplot(X, var_names=None, scatter_kws=None, hist_kws=None,
             grid_kws=None):
    if scatter_kws is None:
        scatter_kws = {}
    if hist_kws is None:
        hist_kws = {}
    if grid_kws is None:
        grid_kws = {
            'b': False,
        }
    n_vars = X.shape[1]
    for i in range(n_vars):
        for j in range(i+1, n_vars):
            for i_, j_ in [[i, j], [j, i]]:
                plt.subplot(n_vars, n_vars, i_*n_vars+j_+1)
                plt.grid(**grid_kws)
                plt.scatter(X[:, j_], X[:, i_], **scatter_kws)
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


def ccf(x, y, lags=40, standardize=True):
    ccf = tsa.ccf(x, y, lags, standardize)
    plt.stem(ccf, use_line_collection=True)
    plt.xlabel('Lags')
    plt.ylabel('CCF')


def acf(x, lags=40, standardize=True):
    acf = tsa.acf(x, lags, standardize)
    plt.stem(acf, use_line_collection=True)
    plt.xlabel('Lags')
    plt.ylabel('ACF')


def pacf(x, lags=40, standardize=True):
    pacf = tsa.pacf(x, lags, standardize)
    plt.stem(pacf, use_line_collection=True)
    plt.xlabel('Lags')
    plt.ylabel('PACF')
