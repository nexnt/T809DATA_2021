import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import scipy.stats as stats
from posixpath import split
from typing import Union

import numpy as np
import sklearn.datasets as datasets

import numpy as np
import sklearn.datasets as datasets
from scipy.stats import multivariate_normal
from sklearn import metrics


def split_train_test(
    features: np.ndarray,
    targets: np.ndarray,
    train_ratio: float = 0.8
) -> Union[tuple, tuple]:
    '''
    Shuffle the features and targets in unison and return
    two tuples of datasets, first being the training set,
    where the number of items in the training set is according
    to the given train_ratio
    '''
    
    p = np.random.permutation(features.shape[0])
    features = features[p]
    targets = targets[p]

    split_index = int(features.shape[0] * train_ratio)

    train_features, train_targets = features[0:split_index, :],\
        targets[0:split_index]
    test_features, test_targets = features[split_index:-1, :],\
        targets[split_index: -1]

    return (train_features, train_targets), (test_features, test_targets)


def load_regression_iris():
    '''
    Load the regression iris dataset that contains N
    input features of dimension F-1 and N target values.

    Returns:
    * features (np.ndarray): A [N x F-1] array of input features
    * targets (np.ndarray): A [N,] array of target values
    '''
    iris = datasets.load_iris()
    return iris.data[:, 0:3], iris.data[:, 3]


# N-dimensional using numpy

from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

'''Class for Gaussian Kernel Regression'''
class GKR:
    
    def __init__(self, x, y, b):
        self.x = np.array(x)
        self.y = np.array(y)
        self.b = b
    
    '''Implement the Gaussian Kernel'''
    def gaussian_kernel(self, z):
        #print((1/np.sqrt(2*np.pi))*np.exp(-0.5*z**2))
        return (1/np.sqrt(2*np.pi))*np.exp(-0.5*z**2)
    
    '''Calculate weights and return prediction'''
    def predict(self, X):
        kernels = np.array([self.gaussian_kernel((np.linalg.norm(xi-X))/self.b) for xi in self.x])
        #print(kernels)
        weights = np.array([len(self.x) * (kernel/np.sum(kernels)) for kernel in kernels])
        #print(weights)
        return np.dot(weights.T, self.y)/len(self.x)
    

X, t = load_regression_iris()

(d_train, t_train), (d_test, t_test) = split_train_test(X, t, train_ratio=0.9)

gkr = GKR(d_train, t_train, 0.09)


pred = []
for d in d_test:
    pred.append(gkr.predict(d))


print(metrics.mean_squared_error(t_test,np.array(pred)))

print(pred)
print(t_test)
