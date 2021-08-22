from math import dist
from sys import intern
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.defchararray import array
from numpy.core.fromnumeric import shape
from numpy.lib.shape_base import dstack
from sklearn.metrics import accuracy_score

from tools import load_iris, split_train_test, plot_points


def euclidian_distance(x: np.ndarray, y: np.ndarray) -> float:
    '''
    Calculate the euclidian distance between points x and y
    '''
    dist = np.linalg.norm(x-y)
    return dist
    ...


def euclidian_distances(x: np.ndarray, points: np.ndarray) -> np.ndarray:
    '''
    Calculate the euclidian distance between x and and many
    points
    '''
    distances = np.zeros(points.shape[0])
    for i in range(points.shape[0]):

        distances[i] = euclidian_distance(x, points[i,:])

    return distances
    ...


def k_nearest(x: np.ndarray, points: np.ndarray, k: int):
    '''
    Given a feature vector, find the indexes that correspond
    to the k-nearest feature vectors in points
    '''
    distances = euclidian_distances(x, points)

    return np.argsort(distances)[0:k]

    ...


def vote(targets, classes):
    '''
    Given a list of nearest targets, vote for the most
    popular
    '''
    count = 0
    winner = int
    
    for i in classes:
        if np.count_nonzero(targets==i) > count:
            count = np.count_nonzero(targets==i)
            winner = i

    return winner

    #np.count_nonzero((self.test_targets == self.guess())[self.test_targets == k])

def knn(
    x: np.ndarray,
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    '''
    Combine k_nearest and vote
    '''
    nearest= k_nearest(x, points, k)
    pointsvote = np.take(point_targets, nearest)
    
    #print(pointsvote, nearest, point_targets)

    return vote(pointsvote, classes)


def knn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:

    prediction = []

    for i in range(len(points)):
        points_2 = remove_one(points, i)
        targets_2 = remove_one(point_targets, i)
        x = points[i,:]
        #print(x)

        currentprediction = knn(x, points_2, targets_2, classes, k)
        
        prediction.append(currentprediction)

    return prediction

    ...

def remove_one(points: np.ndarray, i: int):
    '''
    Removes the i-th from points and returns
    the new array
    '''
    return np.concatenate((points[0:i], points[i+1:]))


def knn_accuracy(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> float:

    knn_predict(points, point_targets, classes, k)

    acc = accuracy_score(point_targets, knn_predict(points, point_targets, classes, k))
    return acc


def knn_confusion_matrix(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:

    predictions = knn_predict(points, point_targets, classes, k)

    
    cm = np.zeros((len(classes), len(classes)))

    for i in classes:
        for k in classes:
            if i==k:
                cm[i,k] = np.count_nonzero((point_targets == predictions)[point_targets == k])
                ...
            else:
                cm[i,k] = np.count_nonzero(np.logical_and(point_targets == k, np.array(predictions) == i))
                ...

    return cm
    
    ...


def best_k(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
) -> int:
    bestkval = int
    accuracy = 0
    interval = range(1,len(points)-1)
    
    print(interval)
    for i in interval:
        acc = knn_accuracy(points, point_targets, classes, i)
        if acc > accuracy:
            accuracy = acc
            bestkval = i
    print(bestkval)
    return bestkval

    
    ...


def knn_plot_points(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
):
    predictions = knn_predict(points, point_targets, classes, k)
    #print(np.array(predictions) == point_targets)
    colors = ['yellow', 'purple', 'blue']
    for i in range(points.shape[0]):
        [x, y] = points[i,:2]
        if predictions[i] == point_targets[i]:
            plt.scatter(x, y, c=colors[point_targets[i]], edgecolors='green', linewidths=2)
        else:
            plt.scatter(x, y, c=colors[point_targets[i]], edgecolors='red', linewidths=2)

    plt.title('Yellow=0, Purple=1, Blue=2')

    plt.show()
    plt.savefig("./02_nearest_neighbours/2_5_1.png")


    ...


def weighted_vote(
    targets: np.ndarray,
    distances: np.ndarray,
    classes: list
) -> int:
    '''
    Given a list of nearest targets, vote for the most
    popular
    '''
    arr = np.stack((targets, distances), axis=-1)
    classsum = np.zeros((len(classes),1))
    for i in classes:
        for k in range(len(arr)):
            if arr[k,0] == i:
                classsum[i] += 1/arr[k,1]

        ...
    '''
    arr = np.stack((targets, distances), axis=-1)
    bla = np.zeros((len(classes),1))
    sum = 0
    for i in range(len(targets)):

        bla[targets[i]] += 1/distances[i]
        sum += bla[targets[i]]
    '''

    return classes[np.argmax(classsum)]

            



def wknn(
    x: np.ndarray,
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    '''
    Combine k_nearest and vote
    '''
    nearest = k_nearest(x, points, k)
    
    pointsvotetarget = np.take(point_targets, nearest)
    pointsvote = np.take(points, nearest, 0)
    distances = euclidian_distances(x, pointsvote)
    
    

    return weighted_vote(pointsvotetarget, distances, classes)


def wknn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    prediction = []

    for i in range(len(points)):
        points_2 = remove_one(points, i)
        targets_2 = remove_one(point_targets, i)
        x = points[i,:]
        #print(x)

        currentprediction = wknn(x, points_2, targets_2, classes, k)
        
        prediction.append(currentprediction)

    return prediction


def compare_knns(
    points: np.ndarray,
    targets: np.ndarray,
    classes: list
):
    acc1 = []
    acc2 = []
    n = []

    for i in range(1,len(points)):
        n.append(i)

        accknn = accuracy_score(targets, knn_predict(points, targets, classes, i))
        accwknn = accuracy_score(targets, wknn_predict(points, targets, classes, i))
        acc1.append(accknn)
        acc2.append(accwknn)

        
    plt.clf()
    plt.plot(n,acc1)
    plt.plot(n,acc2)
    plt.show()
    plt.savefig("./02_nearest_neighbours/b_4_1.png")

d, t, classes = load_iris()
x, points = d[0,:], d[1:, :]
x_target, point_targets = t[0], t[1:]


euclidian_distance(x, points[0])

euclidian_distances(x, points)

k_nearest(x, points, 3)

#vote(np.array([0,0,1,2]), np.array([0,1,2]))
#vote(np.array([1,1,1,1]), np.array([0,1]))


wknn(x, points, point_targets, classes, 5)
#knn(x, points, point_targets, classes, 150)


d, t, classes = load_iris()
(d_train, t_train), (d_test, t_test) = split_train_test(d, t, train_ratio=0.8)

#knn([5.1, 3.4, 1.5, 0.2], remove_one(d_test, 4), point_targets, classes, 1)

predictions = knn_predict(d_test, t_test, classes, 10)
#predictions = knn_predict(d_test, t_test, classes, 5)

knn_accuracy(d_test, t_test, classes, 10)
knn_accuracy(d_test, t_test, classes, 5)

knn_confusion_matrix(d_test, t_test, classes, 10)
knn_confusion_matrix(d_test, t_test, classes, 20)


#best_k(d_train, t_train, classes)

knn_plot_points(d_train, t_train, classes, 3)


compare_knns(d_train, t_train, classes)