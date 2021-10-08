from matplotlib.colors import LinearSegmentedColormap
from numpy.core.fromnumeric import mean
from numpy.lib.function_base import append
import sklearn
from tools import plot_svm_margin, load_cancer
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score)
from matplotlib.offsetbox import AnchoredText
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import timeit
import sklearn.datasets as datasets
from scipy.stats import multivariate_normal
from sklearn import metrics
from typing import Union

import numpy as np
import sklearn.datasets as datasets

def load_regression_diabetes():
    '''
    Load the regression iris dataset that contains N
    input features of dimension F-1 and N target values.

    Returns:
    * features (np.ndarray): A [N x F-1] array of input features
    * targets (np.ndarray): A [N,] array of target values
    '''
    diabetes = datasets.load_diabetes()
    #print(diabetes)
    #print(np.shape(diabetes.data))
    return diabetes.data[:, 0:(np.shape(diabetes.data)[1]-2)], diabetes.data[:, np.shape(diabetes.data)[1]-1]

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


def train_test_SVM(
    svc,
    X_train: np.ndarray,
    t_train: np.ndarray,
    X_test: np.ndarray,
    t_test: np.ndarray,
):
    '''
    Train a configured SVM on <X_train> and <t_train>
    and then measure accuracy, precision and recall on
    the test set

    This function should return (accuracy, precision, recall)
    '''
    start = timeit.default_timer()
    svc.fit(X_train,t_train)
    
    y = svc.predict(X_test)
    end = timeit.default_timer()
    #print(end-start)
    #metrics.mean_squared_error(t_test,y)
    return svc.score(d_test,t_test), end-start



#X, t = load_regression_diabetes()
X, t = load_regression_iris()

(d_train, t_train), (d_test, t_test) = split_train_test(X, t, train_ratio=0.7)
iter=0


for c in np.linspace(0.04, 0.3, num=10):

    res=[]
    for g in np.linspace(0.001, 0.3, num=50):

        msqrs, times = [], []
        for n in range(20):    
            X, t = load_regression_iris()

            (d_train, t_train), (d_test, t_test) = split_train_test(X, t, train_ratio=0.7)
            svc = svm.SVR(kernel='rbf',C=c, gamma=g)
            msqr, time = train_test_SVM(svc, d_train, t_train, d_test, t_test)
            msqrs.append(msqr)
            times.append(time)
            print(c,n)
        res.append([mean(msqrs), mean(times), c, g])
        iter=iter+1
    res=np.array(res)
    plt.plot(res[:,3],res[:,0], label='C = {}'.format(round(c, 3)))

res=np.array(res)

#plt.clf()
plt.title('Relationship between Gamma and Score \nof testdata prediction for different values of C')
#plt.plot(res[:,2],res[:,0])
plt.xlabel("Value of Gamma")
plt.ylabel("Score")
plt.legend(loc='lower right')

plt.savefig("./08_SVM/bonus_1.png")
plt.show()

