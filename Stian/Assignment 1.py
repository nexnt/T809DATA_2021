from typing import Union
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree

from tools import split_train_test


def prior(targets: np.ndarray, classes: list) -> np.ndarray:
    '''
    Calculate the prior probability of each class type
    given a list of all targets and all class types
    '''
    n = targets.shape[0]
    impurities = []
    for c in classes:
        impurities.append(np.where(targets == c)[0].shape[0]/n)
    return np.array(impurities)


def split_data(
    features: np.ndarray,
    targets: np.ndarray,
    split_feature_index: int,
    theta: float
) -> Union[tuple, tuple]:
    '''
    Split a dataset and targets into two seperate datasets
    where data with split_feature < theta goes to 1 otherwise 2
    '''
    features_1 = features[features[:, split_feature_index] < theta, :]
    targets_1 = targets[features[:, split_feature_index] < theta]

    features_2 = features[features[:, split_feature_index] >= theta, :]
    targets_2 = targets[features[:, split_feature_index] >= theta]

    return (features_1, targets_1), (features_2, targets_2)


def gini_impurity(targets: np.ndarray, classes: list) -> float:
    '''
    Calculate:
        i(S_k) = 1/2 * (1 - sum_i P{C_i}**2)
    '''
    return 0.5 * (1 - np.sum(np.power(
        prior(targets, classes),
        2
    )))


def weighted_impurity(
    t1: np.ndarray,
    t2: np.ndarray,
    classes: list
) -> float:
    '''
    Given targets of two branches, return the weighted
    sum of gini branch impurities
    '''
    g1 = gini_impurity(t1, classes)
    g2 = gini_impurity(t2, classes)
    n = t1.shape[0] + t2.shape[0]
    return t1.shape[0]*g1/n + t2.shape[0]*g2/n


def total_gini_impurity(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    split_feature_index: int,
    theta: float
) -> float:
    '''
    Calculate the gini impurity for a split on split_feature_index
    for a given dataset of features and targets.
    '''
    (d1, t1), (d2, t2) = split_data(
        features, targets, split_feature_index, theta)
    return weighted_impurity(t1, t2, classes)


def brute_best_split(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    num_tries: int
) -> Union[float, int, float]:
    '''
    Find the best split for the given data. Test splitting
    on each feature dimension num_tries times.

    Return the lowest gini impurity, the feature dimension and
    the threshold
    '''
    best_gini, best_dim, best_theta = float("inf"), None, None

    # iterate feature dimensions
    for i in range(features.shape[1]):
        # create the thresholds
        thetas = np.linspace(
            features[:, i].min(),
            features[:, i].max(),
            num_tries + 2)[1:-1]
        # iterate thresholds
        for theta in thetas:
            gini = total_gini_impurity(features, targets, classes, i, theta)
            if gini < best_gini:
                best_gini = gini
                best_dim = i
                best_theta = theta
    return best_gini, best_dim, best_theta


class IrisTreeTrainer:
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        classes: list = [0, 1, 2],
        train_ratio: float = 0.8
    ):
        '''
        train_ratio: The ratio of the Iris dataset that will
        be dedicated to training.
        '''
        (self.train_features, self.train_targets),\
            (self.test_features, self.test_targets) =\
            split_train_test(features, targets, train_ratio)

        self.classes = classes
        self.tree = DecisionTreeClassifier()

    def train(self):
        self.tree.fit(self.train_features, self.train_targets)

    def accuracy(self):
        return self.tree.score(self.test_features, self.test_targets)

    def plot(self):
        plot_tree(self.tree)
        plt.show()

    def plot_progress(self):
        train_features = self.train_features
        train_targets = self.train_targets
        accs = []
        for i in range(1, train_features.shape[0]):
            self.train_features = train_features[0:i, :]
            self.train_targets = train_targets[0:i]
            self.train()
            accs.append(self.accuracy())
        plt.plot(accs)
        plt.show()

    def guess(self):
        return self.tree.predict(self.test_features)

    def confusion_matrix(self, features=None, targets=None):
        guesses = self.guess()
        targets = self.test_targets
        matrix = np.zeros([len(self.classes), len(self.classes)])
        for i in range(guesses.shape[0]):
            matrix[guesses[i], targets[i]] += 1
        return matrix

