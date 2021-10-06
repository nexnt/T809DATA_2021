from posixpath import split
from typing import Union

import numpy as np
import sklearn.datasets as datasets
from scipy.stats import multivariate_normal
from sklearn import metrics
from tools import split_train_test

def load_regression_diabetes():
    '''
    Load the regression iris dataset that contains N
    input features of dimension F-1 and N target values.

    Returns:
    * features (np.ndarray): A [N x F-1] array of input features
    * targets (np.ndarray): A [N,] array of target values
    '''
    diabetes = datasets.load_diabetes()
    #print(np.shape(diabetes.data))
    return diabetes.data[:, 0:(np.shape(diabetes.data)[1]-2)], diabetes.data[:, np.shape(diabetes.data)[1]-1]




def mvn_basis(
    features: np.ndarray,
    mu: np.ndarray,
    sigma: float
) -> np.ndarray:
    '''
    Multivariate Normal Basis Function
    The function transforms (possibly many) data vectors <features>
    to an output basis function output <fi>
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional
    data vectors.
    * mu: [MxD] matrix of M D-dimensional mean vectors defining
    the multivariate normal distributions.
    * sigma: All normal distributions are isotropic with sigma*I covariance
    matrices (where I is the MxM identity matrix)
    Output:
    * fi - [NxM] is the basis function vectors containing a basis function
    output fi for each data vector x in features
    '''
    ...
    N, D = features.shape
    M,D = mu.shape
    fi = np.zeros((N,M))
    cov = sigma*np.identity(D)
    for i in range(N):
        for j in range(M):
            fi[i,j] = multivariate_normal.pdf(features[i,:], mean=mu[j,:], cov=cov)
    return fi

def _plot_mvn():
    plt.clf()
    plt.figure(figsize=(15, 10))
    plt.plot(fi)
    
    #plt.show()

    plt.savefig("07_linear_models/1_2_1.png")

def max_likelihood_linreg(
    fi: np.ndarray,
    targets: np.ndarray,
    lamda: float
) -> np.ndarray:
    '''
    Estimate the maximum likelihood values for the linear model

    Inputs :
    * Fi: [NxM] is the array of basis function vectors
    * t: [Nx1] is the target value for each input in Fi
    * lamda: The regularization constant

    Output: [Mx1], the maximum likelihood estimate of w for the linear model
    '''
    N,M = np.shape(fi)
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(fi), fi)+lamda*np.identity(M)),np.transpose(fi)), targets)
    ...


def linear_model(
    features: np.ndarray,
    mu: np.ndarray,
    sigma: float,
    w: np.ndarray
) -> np.ndarray:
    '''
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional data vectors.
    * mu: [MxD] matrix of M D dimensional mean vectors defining the
    multivariate normal distributions.
    * sigma: All normal distributions are isotropic with s*I covariance
    matrices (where I is the MxM identity matrix).
    * w: [Mx1] the weights, e.g. the output from the max_likelihood_linreg
    function.

    Output: [Nx1] The prediction for each data vector in features
    '''
    fi = mvn_basis(features, mu, sigma)
    return np.matmul(fi, np.transpose(w))
    ...


X, t = load_regression_diabetes()

(d_train, t_train), (d_test, t_test) = split_train_test(X, t, train_ratio=0.7)

N, D = d_train.shape

M, sigma, i = 10, 10, 0
mu = np.zeros((M, D))
while i < D:
    #print(X[i, :])
    mmin = np.min(d_train[i, :])
    mmax = np.max(d_train[i, :])
    mu[:, i] = np.linspace(mmin, mmax, M)
    i += 1

#print(mu)

fi = mvn_basis(d_train, mu, sigma)

lamda = 0.01
wml = max_likelihood_linreg(fi, t_train, lamda)
print(linear_model(d_test, mu, sigma, wml))

print(t_test)

print(metrics.mean_squared_error(linear_model(d_test, mu, sigma, wml),t_test))
