from typing import Union
import numpy as np

from tools import load_iris, split_train_test


def sigmoid(x: float) -> float:
    '''
    Calculate the sigmoid of x
    '''
    if x<-100:
        return 0.0
    else:
        return 1/(1+np.exp(-x))
    ...


def d_sigmoid(x: float) -> float:
    '''
    Calculate the derivative of the sigmoid of x.
    '''
    return np.exp(-x)/np.power(1+np.exp(-x),2)
    ...


def perceptron(
    x: np.ndarray,
    w: np.ndarray
) -> Union[float, float]:
    '''
    Return the weighted sum of x and w as well as
    the result of applying the sigmoid activation
    to the weighted sum
    '''
    res = np.dot(x,w)
    return res, sigmoid(res)
    ...


def ffnn(
    x: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray,
) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Computes the output and hidden layer variables for a
    single hidden layer feed-forward neural network.
    '''

    a1 = np.zeros((1,M))
    a2 = np.zeros((1,K))

    z0 = np.zeros((1,len(x)+1))
    z1 = np.zeros((1,M+1))
    y = np.zeros((1,K))

    return y, z0, z1, a1, a2
    ...



def backprop(
    x: np.ndarray,
    target_y: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray
) -> Union[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Perform the backpropagation on given weights W1 and W2
    for the given input pair x, target_y
    '''
    ...


def train_nn(
    X_train: np.ndarray,
    t_train: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray,
    iterations: int,
    eta: float
) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Train a network by:
    1. forward propagating an input feature through the network
    2. Calculate the error between the prediction the network
    made and the actual target
    3. Backpropagating the error through the network to adjust
    the weights.
    '''
    ...


def test_nn(
    X: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray
) -> np.ndarray:
    '''
    Return the predictions made by a network for all features
    in the test set X.
    '''
    ...


#print(sigmoid(0.5))
#print(d_sigmoid(0.2))

#print(perceptron(np.array([1.0, 2.3, 1.9]),np.array([0.2,0.3,0.1])))

features, targets, classes = load_iris()
(train_features, train_targets), (test_features, test_targets) = \
    split_train_test(features, targets)

# Take one point:
x = train_features[0, :]
K = 3 # number of classes
M = 10
D=len(train_features[0, :])
# Initialize two random weight matrices
W1 = 2 * np.random.rand(D + 1, M) - 1
W2 = 2 * np.random.rand(M + 1, K) - 1
y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)