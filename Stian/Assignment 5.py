from typing import Union
import numpy as np

from tools import load_iris, split_train_test


def sigmoid(x: float) -> float:
    '''
    Calculate the sigmoid of x
    '''
    if x < -100:
        return 0.0
    else:
        return 1 / (1 + np.exp(-x))


def d_sigmoid(x: float) -> float:
    '''
    Calculate the derivative of the sigmoid of x.
    '''
    xx = sigmoid(x)
    return xx * (1 - xx)


def perceptron(
    x: np.ndarray,
    w: np.ndarray
) -> Union[float, float]:
    '''
    Return the weighted sum of x and w as well as
    the result of applying the sigmoid activation
    to the weighted sum
    '''
    a = np.dot(x, w)
    return a, sigmoid(a)


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

    z0 = np.insert(x, 0, 1.0)

    z1 = np.zeros(M)
    a1 = np.zeros(M)

    a2 = np.zeros(K)
    y = np.zeros(K)

    for m in range(M):
        a1[m], z1[m] = perceptron(z0, W1[:, m])

    z1 = np.insert(z1, 0, 1.0)

    for k in range(K):
        a2[k], y[k] = perceptron(z1, W2[:, k])

    return y, z0, z1, a1, a2


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
    # perform the forward pass
    y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)

    delta_k = y - target_y
    delta_j = np.zeros(M+1)

    # calculate delta_j
    for m in range(M):
        total = 0.0
        for k in range(K):
            total += delta_k[k]*W2[m+1, k]
        delta_j[m] = d_sigmoid(a1[m]) * total

    # initialize weight error gradient matrices
    dE1 = np.zeros(W1.shape)
    dE2 = np.zeros(W2.shape)

    # Calculate the gradient for W1
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            dE1[i, j] = delta_j[j]*z0[i]

    # Calculate the gradient for W2
    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            dE2[i, j] = delta_k[j]*z1[i]

    return y, dE1, dE2


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
    N, D = X_train.shape
    Etotal = np.zeros(iterations)
    misclassification_rate = np.zeros(iterations)

    for iteration in range(iterations):
        correct_guesses = 0
        Ee = 0
        dE1_total = np.zeros(W1.shape)
        dE2_total = np.zeros(W2.shape)

        guesses = np.zeros(N)

        for i in range(N):
            target_y = np.zeros(K)
            target_y[t_train[i]] = 1
            x = X_train[i, :]

            y, dE1, dE2 = backprop(x, target_y, M, K, W1, W2)
            dE1_total = dE1_total + dE1
            dE2_total = dE2_total + dE2

            if np.argmax(y) == t_train[i]:
                correct_guesses += 1
            guesses[i] = np.argmax(y)

            for ik in range(K):
                Ee = Ee - target_y[ik] * np.log(y[ik])\
                    - (1 - target_y[ik]) * np.log(1 - y[ik])

        Etotal[iteration] = Ee / N
        misclassification_rate[iteration] = 1 - correct_guesses / N

        W1 = W1 - eta * dE1_total / N
        W2 = W2 - eta * dE2_total / N

    return W1, W2, Etotal, misclassification_rate, guesses


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
    N, D = np.shape(X)

    guesses = np.zeros(N)
    for i in range(N):
        x = X[i, :]
        y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)
        guesses[i] = np.argmax(y)
    return guesses
