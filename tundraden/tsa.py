import numpy as np
import scipy.signal
import scipy.linalg

from . import utils


def generate_arma_samples(ar, ma, n_samples, sigma=1):
    '''
    Generate random samples from an ARMA process

    Parameters:
        ar:
            Autoregressive coefficients, zero-lag included. `ar[0]` must be 1.
        ma:
            Moving average coefficients, zero-lag included. `ma[0]` must be 1.
        n:
            Number of samples to generate.
        sigma:
            Noise standard deviation.

    Returns:
        samples:
            Array of samples of length `n_samples`.
    '''
    assert ar[0] == 1
    assert ma[0] == 1
    noise = sigma*np.random.randn(n_samples)
    return scipy.signal.lfilter(ma, ar, noise)


def generate_varma_samples(AR, MA, n_samples, Sigma=None):
    '''
    Generate random samples from an VARMA process

    Parameters:
        AR:
            List of autoregressive matrices, zero-lag included. All matrices in
            `AR` and `MA` must have the same dimension. `AR[0]` must be the
            identity matrix.
        MA:
            List of moving average matrices, zero-lag included. All matrices in
            `AR` and `MA` must have the same dimension. `MA[0]` must be the
            identity matrix.
        n:
            Number of samples to generate.
        Sigma:
            Noise covariance matrix. Must have the same dimensions as the
            matrices in AR and MA.

    Returns:
        samples:
            Array of samples with size `n_samples*n_variables`.
    '''
    assert all([ar.ndim == 2 for ar in AR])
    assert all([ma.ndim == 2 for ma in MA])
    assert all([ar.shape[0] == ar.shape[1] for ar in AR])
    assert all([ma.shape[0] == ma.shape[1] for ma in MA])
    assert all([ar.shape[0] == AR[0].shape[0] for ar in AR])
    assert all([ma.shape[0] == MA[0].shape[0] for ma in MA])
    assert (AR[0] == np.identity(AR[0].shape[0])).all()
    assert (MA[0] == np.identity(MA[0].shape[0])).all()

    if Sigma is not None:
        assert Sigma.ndim == 2
        assert Sigma.shape[0] == Sigma.shape[1]
        assert Sigma.shape[0] == AR[0].shape[0]
        assert Sigma.shape[0] == MA[0].shape[0]
    else:
        Sigma = np.identity(AR[0].shape[0])

    n_variables = Sigma.shape[0]
    mu = np.zeros(n_variables)
    noise = np.random.multivariate_normal(mu, Sigma, size=n_samples)
    samples = np.zeros((n_samples, n_variables))
    for i in range(n_samples):
        for j, ar in enumerate(AR):
            if i-j >= 0:
                samples[i, :] -= np.dot(ar, samples[i-j, :])
        for j, ma in enumerate(MA):
            if i-j >= 0:
                samples[i, :] += np.dot(ma, noise[i-j, :])
    return samples


def ccf(x, y, lags=40, standardize=True):
    '''
    Cross-correlation function or cross-covariance function.

    Parameters:
        x:
            First time series i.e. the one the delay is applied to.
        y:
            Second time series.
        lags:
            Maximum lag to compute the cross-correlation for.
        standardize:
            Wether to standardize the series before the cross-covariance and
            variances calculation. You should set this to `False` only if you
            know the data comes from a zero-mean and unit-variance
            distribution.

    Returns:
        ccf:
            Cross-correlation or cross-covariance values. Length `lags+1`.
    '''
    if standardize:
        x = utils.standardize(x)
        y = utils.standardize(y)
    ccf = np.zeros(min(len(y), lags+1))
    n = max(len(x), len(y))
    for i in range(len(ccf)):
        y_advanced = y[i:]
        m = min(len(x), len(y_advanced))
        ccf[i] = np.sum(x[:m]*y_advanced[:m])
    ccf /= n
    return ccf


def acf(x, lags=40, standardize=True):
    '''
    Autocorrelation function. Matches default behavior of its R equivalent.

    Parameters:
        x:
            The time series.
        lags:
            Maximum lag to compute the autocorrelation for.
        standardize:
            Wether to standardize the series before the cross-covariance and
            variances calculation. You should set this to `False` only if you
            know the data comes from a zero-mean and unit-variance
            distribution.

    Returns:
        acf:
            Autocorrelation values. Length `lags+1`.
    '''
    return ccf(x, x, lags, standardize)


def pacf(x, lags=40, standardize=True):
    '''
    Partial autocorrelation function. Matches default behavior of its R
    equivalent.

    Inspired from https://stats.stackexchange.com/a/129374. Probably
    inefficient. The Levinson recursion might be better.
    '''
    pacf = np.zeros(min(lags, len(x)-1))
    acf_ = acf(x, lags, standardize)
    R = scipy.linalg.toeplitz(acf_[:-1])
    for i in range(len(pacf)):
        pacf[i] = scipy.linalg.solve(R[:i+1, :i+1], acf_[1:i+2])[-1]
    return pacf
