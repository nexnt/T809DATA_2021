from typing import Union
from warnings import resetwarnings
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.defchararray import array
from sklearn.tree import DecisionTreeClassifier, plot_tree

from tools import load_iris, split_train_test


def prior(targets: np.ndarray, classes: list) -> np.ndarray:
    '''
    Calculate the prior probability of each class type
    given a list of all targets and all class types
    '''
    count = 0
    result = []

    for i in range(len(classes)):
        count = 0
        for n in range(len(targets)):
            
            if targets[n] == classes[i] :
                count +=1

        result.append(count/len(targets))
    
    #print(result)
    
    return result
    ...


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
    features_1 = []
    features_2 = []
    targets_1 = []
    targets_2 = []

    for i in range(len(features)):
        #print(features[i, split_feature_index-1])

        if features[i, split_feature_index] < theta:
            features_1.append(features[i])
            targets_1.append(targets[i])
            
        else:
            features_2.append(features[i])
            targets_2.append(targets[i])
  


    #print(features_1.shape[0])
    features_1 = np.array(features_1)
    features_2 = np.array(features_2)

    return (features_1, targets_1), (features_2, targets_2)


def gini_impurity(targets: np.ndarray, classes: list) -> float:
    '''
    Calculate:
        i(S_k) = 1/2 * (1 - sum_i P{C_i}**2)
    '''
    if len(targets) > 0:
        sum = 0
        for i in classes:
            count = np.count_nonzero(np.array(targets) == i)/len(targets)
            sum += count**2
        return 1/2*(1-sum)
    else:
        return 1
    
    
       
    ...


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
    n = len(t1) + len(t2)

    result = len(t1)*g1/n+len(t2)*g2/n

    return result
    ...


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

    (f_1, t_1), (f_2, t_2) = split_data(features, targets, split_feature_index, theta)

    return weighted_impurity(t_1, t_2, classes)




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
    last_gini = 0.5

    best_gini, best_dim, best_theta = float("inf"), None, None
    # iterate feature dimensions
    for i in range(features.shape[1]):
        # create the thresholds
        min_value = min(features[:,i]) #+(0.05*(max(features[:,i])-min(features[:,i])))
        max_value = max(features[:,i]) #-(0.05*(max(features[:,i])-min(features[:,i])))

        thetas = np.linspace(min_value, max_value, num_tries+2)[1:-1]

        #thetas = ...
        # iterate thresholds

        for theta in thetas:
            #gini = total_gini_impurity(features, targets, classes, 0, 1)
            current_gini = total_gini_impurity(features, targets, classes, i, theta)
            if current_gini < last_gini:
                last_gini = current_gini
                best_gini = current_gini
                best_dim = i
                best_theta = theta

    #print(array(results))
    
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
        self.features = features
        self.targets = targets
        self.tree = DecisionTreeClassifier()

    def train(self):
        return self.tree.fit(self.features, self.targets)

    def accuracy(self):
        ...

    def plot(self):
        plot_tree
        plt.show()
        plt.savefig("./01_decision_trees/2_3_1.png")

    def plot_progress(self):
        # Independent section
        # Remove this method if you don't go for independent section.
        ...

    def guess(self):
        return self.tree.predict(self.features)

    def confusion_matrix(self):
        ...


#prior([0, 0, 1], [0, 1])

features, targets, classes = load_iris()

#(f_1, t_1), (f_2, t_2) = split_data(features, targets, 2, 4.65)

#gini_impurity(t_1, classes)

#weighted_impurity(t_1, t_2, classes)

#total_gini_impurity(features, targets, classes, 2, 4.65)
#total_gini_impurity(features, targets, classes, 0, 1)
#print(brute_best_split(features, targets, classes, 30))

#IrisTreeTrainer(features, targets, classes)

features, targets, classes = load_iris()
dt = IrisTreeTrainer(features, targets, classes=classes)
dt.train()
#print(f'The accuracy is: {dt.accuracy()}')
dt.plot()
print(f'I guessed: {dt.guess()}')
#print(f'The true targets are: {dt.t_test}')
#print(dt.confusion_matrix())