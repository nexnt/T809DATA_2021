from tools import scatter_3d_data, bar_per_axis

import matplotlib.pyplot as plt
import numpy as np


def gen_data(
    n: int,
    k: int,
    mean: np.ndarray,
    var: float
) -> np.ndarray:
    '''Generate n values samples from the k-variate
    normal distribution
    '''
    assert mean.shape[0] == k,\
        "mean dimension does not correspond to k"

    covar = np.power(var, 2)*np.identity(k, dtype=float)
    return np.random.multivariate_normal(mean, covar, n)


def update_sequence_mean(
    mu: np.ndarray,
    x: np.ndarray,
    n: int
) -> np.ndarray:
    '''Performs the mean sequence estimation update
    '''
    print(mu + (x-mu)/n)
    return mu + (x-mu)/n


def _plot_sequence_estimate():
    data = gen_data(100, 3, np.array([0, 0, 0]), 4)
    estimates = [np.array([0, 0, 0])]
    for i in range(data.shape[0]):
        estimates.append(update_sequence_mean(estimates[i], data[i, :], i+1))
    plt.plot([e[0] for e in estimates], label='First dimension')
    plt.plot([e[1] for e in estimates], label='Second dimension')
    plt.plot([e[2] for e in estimates], label='Third dimension')
    plt.legend(loc='upper center')
    plt.show()

def _square_error(y, y_hat):
    return np.square(y-y_hat).mean(axis=0)

def _plot_mean_square_error():
    data = gen_data(100, 3, np.array([2, -2, 3]), np.sqrt(10))
    estimates = [np.array([0, 0, 0])]
    errors = [_square_error(estimates[-1], np.array([2, -2, 3]))]
    for i in range(data.shape[0]):
        estimates.append(update_sequence_mean(estimates[i], data[i, :], i+1))
        errors.append(_square_error(estimates[-1], np.array([2, -2, 3])))
    plt.plot(errors)
    plt.show()


# Naive solution to the bonus question.

def gen_changing_data(
    n: int,
    k: int,
    start_mean: np.ndarray,
    end_mean: np.ndarray,
    var: float
) -> np.ndarray:

    means = np.linspace(start_mean, end_mean, n)
    covar = np.power(var, 2)*np.identity(k, dtype=float)
    data = np.zeros([n, k])
    # sample n times from n different means
    for i in range(n):
        data[i, :] = np.random.multivariate_normal(means[i, :], covar, 1)
    return data

def _plot_changing_sequence_estimate():
    data = gen_changing_data(1000, 3, np.array([0, 0, 0]), np.array([5, 5, 5]), np.sqrt(3))
    seen_data = np.zeros([1000, 3])
    estimates = []
    for i in range(data.shape[0]):
        seen_data[i, :] = data[i, :]
        # take a rolling average with a window of twenty:
        rolling_average = np.mean(seen_data[max(0, i-20):i, :], axis=0)
        estimates.append(rolling_average)
    plt.plot([e[0] for e in estimates], label='First dimension')
    plt.plot([e[1] for e in estimates], label='Second dimension')
    plt.plot([e[2] for e in estimates], label='Third dimension')
    plt.legend(loc='upper center')
    plt.show()


_plot_changing_sequence_estimate()