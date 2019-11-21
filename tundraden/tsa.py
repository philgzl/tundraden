import numpy as np
import scipy


def generate_arma_samples(ar, ma, n, sigma=1):
    noise = sigma*np.random.randn(n)
    return scipy.signal.lfilter(ma, ar, noise)


def acf(x, lags=40):
    output = np.zeros(min(lags+1, len(x)))
    output[0] = np.sum((x-x.mean())**2)
    for i in range(1, len(output)):
        x1 = x[i:]
        x2 = x[:-i]
        output[i] = np.sum((x1-x.mean())*(x2-x.mean()))
    output /= output[0]
    return output


def pacf(x, lags=40):
    '''
    Probably extremely inefficient. The Levinson recursion might be better.
    '''
    output = np.zeros(min(lags, len(x)-1))
    acf_ = acf(x, lags)
    R = scipy.linalg.toeplitz(acf_[:-1])
    for i in range(len(output)):
        output[i] = scipy.linalg.solve(R[:i+1, :i+1], acf_[1:i+2])[-1]
    return output
