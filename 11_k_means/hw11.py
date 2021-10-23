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
    distances = np.zeros((n,k))
    for i in range(n):
        for j in range(k):
            distances[i,j] = np.sqrt(sum(np.power(X[i,:] - Mu[j,:],2)))
    return distances


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
    for n in range(dist.shape[0]):
        for k in range(dist.shape[1]):
            J[n, k] = R[n, k] * dist[n, k]

    res = sum(sum(J))/J.shape[0]
    return res


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

    N, K = R.shape
    newmu = np.zeros_like(Mu)
    for k in range(K):
        res1=0
        res2=0
        for n in range(N):
            res1 += R[n,k]*X[n]
            res2 += R[n,k]
        if res2 != 0:
            newmu[k] = res1/res2
        else:
            newmu[k] = Mu[k]
    return newmu
    
    ...


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
        distances = distance_matrix(X_standard, Mu)
        R = determine_r(distances)
        J = determine_j(R, distances)
        Js.append(J)
        Mu = update_Mu(Mu, X_standard, R)
        ...

    # Then we have to "de-standardize" the prototypes
    for i in range(k):
        Mu[i, :] = Mu[i, :] * X_std + X_mean

    return Mu, R, Js


def _plot_j():

    Mu, R, Js = k_means(X, 4, 10)
    plt.clf
    plt.plot(Js)
    plt.title("Evolution of J")
    plt.ylabel("Value of J")
    plt.xlabel("Iteration")
    plt.savefig("11_k_means/1_6_1.png")
    plt.show()


def _plot_multi_j():
    K= [2,3,5,10]
    plt.clf 
    for k in K:
        Mu, R, Js = k_means(X, k, 10)
        plt.plot(Js, label='k = {}'.format(k))
    plt.title("Evolution of J")    
    plt.ylabel("Value of J")
    plt.xlabel("Iteration")
    plt.legend(loc='upper center')
    plt.savefig("11_k_means/1_7_1.png")


    plt.show()

def k_means_predict(
    X: np.ndarray,
    t: np.ndarray,
    classes: list,
    num_its: int
) -> np.ndarray:
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
    * the predictions (list)
    '''
    nc = len(classes)
    nt = len(t)
    Mu, R, Js = k_means(X, k=nc, num_its=num_its)
    conv = np.zeros([nc, nc])
    for n in range(nt):
        conv[t[n], R[n, :].argmax()] += 1
        cl2=[]
    for n in range(nt):
        cl2.append(R[n, :].argmax())
    pred=[]
    for i in cl2:
        pred.append(conv[:, i].argmax())
    return pred

def _iris_kmeans_accuracy():
    y_pred = k_means_predict(X, y, c, 5)
    print(accuracy_score(y,y_pred))
    print(confusion_matrix(y,y_pred))

def _my_kmeans_on_image():
    ...


def plot_image_clusters(n_clusters: int):
    '''
    Plot the clusters found using sklearn k-means.
    '''
    image, (w, h) = image_to_numpy()
    c = KMeans(n_clusters=n_clusters)
    c.fit(image)
    print(w, h)

    plt.subplot('121')
    plt.title("Original image")
    plt.imshow(image.reshape(w, h, 3))
    plt.ylabel("Pixels")
    plt.xlabel("Pixels")
    plt.subplot('122')
    
    plt.title("Plot using {} clusters".format(n_clusters))
    # uncomment the following line to run
    plt.xlabel("Pixels")
    plt.imshow(c.labels_.reshape(w, h), cmap="plasma")
    plt.savefig("11_k_means/2_1_{}.png".format(n_clusters))
    plt.show()


def _gmm_info():
    c = GaussianMixture(n_components=3)
    c.fit(X)
    print(c.means_)
    print(c.covariances_)
    print(c.weights_)

def _plot_gmm():
    c = GaussianMixture(n_components=3)
    c.fit(X)
    

    plot_gmm_results(X,c.predict(X),c.means_,c.covariances_)

X, y, c = load_iris()
#print(k_means(X, 4, 10))
#_plot_gmm()

#_plot_j()
#_plot_multi_j()

#print(k_means_predict(X, y, c, 5))

#_iris_kmeans_accuracy()


#plot_image_clusters(20)

_gmm_info()