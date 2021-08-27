import numpy as np
import matplotlib.pyplot as plt

from tools import load_iris, split_train_test, plot_points

def euclidian_distance(x: np.ndarray, y: np.ndarray) -> float:
    '''
    Calculate the euclidian distance between points x and y
    '''
    return np.sqrt(np.sum(np.square(x-y)))


def euclidian_distances(x: np.ndarray, points: np.ndarray) -> np.ndarray:
    '''
    Calculate the euclidian distance between x and and many
    points
    '''
    distances = np.zeros(points.shape[0])
    for i in range(points.shape[0]):
        distances[i] = euclidian_distance(x, points[i, :])
    return distances


def k_nearest(x: np.ndarray, points: np.ndarray, k: int):
    '''
    Given a feature vector, find the indexes that correspond
    to the k-nearest feature vectors in points
    '''
    distances = euclidian_distances(x, points)
    return np.argsort(distances)[:k]


def vote(targets, classes):
    '''
    Given a list of nearest targets, vote for the most
    popular
    '''
    votes = np.zeros(len(classes))
    for i in range(targets.shape[0]):
        votes[targets[i]] += 1
    return np.argmax(votes)


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
    near_idx = k_nearest(x, points, k)
    near_targets = point_targets[near_idx]
    return vote(near_targets, classes)


def knn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    predictions = np.empty([points.shape[0]], dtype=int)
    for i in range(points.shape[0]):
        predictions[i] = knn(
            points[i, :],
            np.concatenate((points[0:i, :], points[i+1:, :])),
            np.concatenate((point_targets[0:i], point_targets[i+1:])),
            classes,
            k)
    return predictions


def knn_accuracy(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> float:
    predictions = knn_predict(points, point_targets, classes, k)
    return np.where(predictions == point_targets)[0].shape[0] /\
        point_targets.shape[0]


def knn_confusion_matrix(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    predictions = knn_predict(points, point_targets, classes, k)
    confusion = np.zeros([len(classes), len(classes)])
    for i in range(predictions.shape[0]):
        confusion[predictions[i], point_targets[i]] += 1
    return confusion


def best_k(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
) -> int:
    accs = np.zeros([points.shape[0]])
    for k in range(1, points.shape[0] - 1):
        accs[k-1] = knn_accuracy(points, point_targets, classes, k)
    return np.argmax(accs) + 1


def knn_plot_points(points, point_targets, classes, k):
    predictions = knn_predict(points, point_targets, classes, k)
    colors = ['yellow', 'purple', 'blue']
    for i in range(points.shape[0]):
        [x, y] = points[i, :2]
        if point_targets[i] == predictions[i]:
            edge_color = 'g'
        else:
            edge_color = 'r'
        plt.scatter(
            x,
            y,
            c=colors[point_targets[i]],
            edgecolors=edge_color,
            linewidths=2)
    plt.title('Yellow=0, Purple=1, Blue=2')
    plt.show()


def weighted_vote(targets, distances, classes):
    '''
    Given a list of nearest targets, vote for the most
    popular
    '''
    # we add a small value to distances in case the same
    # feature vector appears twice in the dataset
    weights = 1.0/(distances+np.finfo(np.float).eps)
    weight_sum = np.sum(weights)
    votes = np.zeros(len(classes))
    for i in range(targets.shape[0]):
        votes[targets[i]] += weights[i]/weight_sum
    return np.argmax(votes)


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
    near_idx = k_nearest(x, points, k)
    distances = euclidian_distances(x, points)
    near_targets = point_targets[near_idx]
    near_distances = distances[near_idx]
    return weighted_vote(near_targets, near_distances, classes)


def wknn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    predictions = np.empty([points.shape[0]], dtype=int)
    for i in range(points.shape[0]):
        predictions[i] = wknn(
            points[i, :],
            np.concatenate((points[0:i, :], points[i+1:, :])),
            np.concatenate((point_targets[0:i], point_targets[i+1:])),
            classes,
            k)
    return predictions


def compare_knns(points, targets, classes):
    k_acc, w_acc = [], []
    for k in range(1, points.shape[0]):
        w_prediction = wknn_predict(points, targets, classes, k)
        k_prediction = knn_predict(points, targets, classes, k)
        k_acc.append(
            np.where(k_prediction == targets)[0].shape[0]/points.shape[0])
        w_acc.append(
            np.where(w_prediction == targets)[0].shape[0]/points.shape[0])

    plt.plot(range(1, points.shape[0]), k_acc, label='knn')
    plt.plot(range(1, points.shape[0]), w_acc, label='wknn')
    plt.legend(loc='bottom left')
    plt.show()
