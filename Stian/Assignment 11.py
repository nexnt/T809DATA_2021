import numpy as np
import sklearn as sk
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from typing import Union

from tools import load_iris, image_to_numpy, plot_gmm_results


def distance_matrix(
    X: np.ndarray,
    Mu: np.ndarray
) -> np.ndarray:
    '''
    Returns a matrix of euclidian distances between points in
    X and Mu.

    Input arguments:
    * X (np.ndarray): A [n x f] array of samples
    * Mu (np.ndarray): A [k x f] array of prototypes

    Returns:
    out (np.ndarray): A [n x k] array of euclidian distances
    where out[i, j] is the euclidian distance between X[i, :]
    and Mu[j, :]
    '''


    n,f = X.shape
    k,f = Mu.shape
    dist = np.zeros((n,k))
    for i in range(n):
        for j in range(k):
            dist[i,j] = sum(np.power(X[i] - Mu[j],2))
    return np.sqrt(dist)


def determine_r(dist: np.ndarray) -> np.ndarray:
    '''
    Returns a matrix of binary indicators, determining
    assignment of samples to prototypes.

    Input arguments:
    * dist (np.ndarray): A [n x k] array of distances

    Returns:
    out (np.ndarray): A [n x k] array where out[i, j] is
    1 if sample i is closest to prototype j and 0 otherwise.
    '''
    R = np.zeros((dist.shape))
    for i in range(dist.shape[0]):
        R[i,np.argmin(dist[i,:])] = 1
    return R



def determine_j(R: np.ndarray, dist: np.ndarray) -> float:
    '''
    Calculates the value of the objective function given
    arrays of indicators and distances.

    Input arguments:
    * R (np.ndarray): A [n x k] array where out[i, j] is
        1 if sample i is closest to prototype j and 0
        otherwise.
    * dist (np.ndarray): A [n x k] array of distances

    Returns:
    * out (float): The value of the objective function
    '''
    
    J = np.zeros((dist.shape))
    for i in range(dist.shape[0]):
        for n in range(dist.shape[1]):
            J[i, n] = R[i, n] * dist[i, n]

            
    return sum(sum(J))/J.shape[0]

def update_Mu(
    Mu: np.ndarray,
    X: np.ndarray,
    R: np.ndarray
) -> np.ndarray:
    '''
    Updates the prototypes, given arrays of current
    prototypes, samples and indicators.

    Input arguments:
    Mu (np.ndarray): A [k x f] array of current prototypes.
    X (np.ndarray): A [n x f] array of samples.
    R (np.ndarray): A [n x k] array of indicators.

    Returns:
    out (np.ndarray): A [k x f] array of updated prototypes.
    '''
    K, F = Mu.shape
    N, F = X.shape
    upd_mu = np.zeros_like(Mu)

    for k in range(K):
        sum_up = 0
        sum_down = 0
        for n in range(N):
            sum_up += R[n,k]*X[n]
            sum_down += R[n,k]
        if sum_down == 0:
            sum_down = Mu[k]    
        upd_mu[k] = sum_up / sum_down

    return upd_mu


def k_means(
    X: np.ndarray,
    k: int,
    num_its: int
) -> Union[list, np.ndarray, np.ndarray]:
    # We first have to standardize the samples
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_standard = (X-X_mean)/X_std
    # run the k_means algorithm on X_st, not X.

    # we pick K random samples from X as prototypes
    nn = sk.utils.shuffle(range(X_standard.shape[0]))
    Mu = X_standard[nn[0: k], :]
    Js = []

    for i in range(num_its):
        dist = distance_matrix(X_standard, Mu)
        R = determine_r(dist)
        J = determine_j(R, dist)
        Js.append(J)
        Mu = update_Mu(Mu, X_standard, R)

    # Then we have to "de-standardize" the prototypes
    for i in range(k):
        Mu[i, :] = Mu[i, :] * X_std + X_mean

    return Mu, R, Js

def _plot_j():
    a, b, c = k_means(X, 4, 10)
    plt.plot(c)


def _plot_multi_j():
    a, b, c1 = k_means(X, 2, 10)
    plt.plot(c1)
    a, b, c2 = k_means(X, 3, 10)
    plt.plot(c2)
    a, b, c3 = k_means(X, 5, 10)
    plt.plot(c3)
    a, b, c4 = k_means(X, 10, 10)
    plt.plot(c4)


def k_means_predict(
    X: np.ndarray,
    t: np.ndarray,
    classes: list,
    num_its: int
) -> Union[float, np.ndarray]:
    '''
    Determine the accuracy and confusion matrix
    of predictions made by k_means on a dataset
    [X, t] where we assume the most common cluster
    for a class label corresponds to that label.

    Input arguments:
    * X (np.ndarray): A [n x f] array of input features
    * t (np.ndarray): A [n] array of target labels
    * classes (list): A list of possible target labels
    * num_its (int): Number of k_means iterations

    Returns:
    * accuracy (float): The accuracy of the prediction
    * confusion_matrix (np.ndarray): The confusion matrix
        of the prediction and the ground truth labels.
    '''
    N, cc = len(t), len(classes)
    _, R, _ = k_means(X, cc, num_its)
    mest = np.zeros([cc, cc])
    for n in range(N):
        mest[t[n], R[n, :].argmax()] += 1
    
    preds = [None]*N
    for i in range(N):
        r_max = R[i, :].argmax()
        preds[i] = mest[:, r_max].argmax()
    return preds


def _iris_kmeans_accuracy():
    y_pred = k_means_predict()
    y_true = k_means()
    return accuracy_score(y_true, y_pred)


def _my_kmeans_on_image():
    ...


def plot_image_clusters(n_clusters: int):
    '''
    Plot the clusters found using sklearn k-means.
    '''
    image, (w, h) = image_to_numpy()
    ...
    plt.subplot('121')
    plt.imshow(image.reshape(w, h, 3))
    plt.subplot('122')
    # uncomment the following line to run
    # plt.imshow(kmeans.labels_.reshape(w, h), cmap="plasma")
    plt.show()


def _gmm_info():
    ...


def _plot_gmm():
    ...

