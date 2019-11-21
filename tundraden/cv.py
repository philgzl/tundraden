import numpy as np


def train_test_split(X, Y, train_size=None, test_size=None, shuffle=True):
    if train_size is not None and test_size is not None:
        raise ValueError(''''train_size' and 'test_size' can't be both
                         specified''')
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
        raise ValueError('''Either 'train_size' or 'test_size' should be
                         specified''')
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
