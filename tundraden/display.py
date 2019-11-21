import matplotlib.pyplot as plt

from . import tsa


def pairplot(X, var_names=None, scatter_kws=None, hist_kws=None,
             grid_kws=None):
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


def acf(x, lags=40):
    acf_ = tsa.acf(x, lags)
    plt.stem(acf_, use_line_collection=True)


def pacf(x, lags=40):
    pacf_ = tsa.pacf(x, lags)
    plt.stem(pacf_, use_line_collection=True)
