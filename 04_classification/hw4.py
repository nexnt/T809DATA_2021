from re import A
from numpy.core.fromnumeric import shape
from numpy.lib.function_base import append
from numpy.ma.core import concatenate
from tools import load_iris, split_train_test

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def mean_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the mean of a selected class given all features
    and targets in a dataset
    '''
    return np.mean(features[targets == selected_class,:], axis=0)
    ...


def covar_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the covariance of a selected class given all
    features and targets in a dataset
    '''
    return np.cov(features[targets == selected_class,:], rowvar=False)
    ...


def likelihood_of_class(
    feature: np.ndarray,
    class_mean: np.ndarray,
    class_covar: np.ndarray
) -> float:
    '''
    Estimate the likelihood that a sample is drawn
    from a multivariate normal distribution, given the mean
    and covariance of the distribution.
    '''
    return multivariate_normal(mean=class_mean, cov=class_covar).pdf(feature)
    ...


def maximum_likelihood(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum likelihood for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''
    means, covs = [], []
    for class_label in classes:
        means.append(mean_of_class(train_features, train_targets, class_label))

        covs.append(covar_of_class(train_features, train_targets, class_label))

        ...
    likelihoods = []
    for i in range(test_features.shape[0]):
        likelihood = np.zeros(len(classes))

        for c in range(len(classes)):
            likelihood[c] = likelihood_of_class(test_features[i,:], means[c], covs[c])

        ...
        likelihoods.append(likelihood)
    return np.array(likelihoods)


def predict(likelihoods: np.ndarray):
    '''
    Given an array of shape [num_datapoints x num_classes]
    make a prediction for each datapoint by choosing the
    highest likelihood.

    You should return a [likelihoods.shape[0]] shaped numpy
    array of predictions, e.g. [0, 1, 0, ..., 1, 2]
    '''
    predictions = np.empty(likelihoods.shape[0])
    for i in range(likelihoods.shape[0]):
        predictions[i] = (np.argmax(likelihoods[i,:]))
    return predictions
    ...


def maximum_aposteriori(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum a posteriori for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''
    means, covs, apos = [], [], []
    for class_label in classes:
        means.append(mean_of_class(train_features, train_targets, class_label))

        covs.append(covar_of_class(train_features, train_targets, class_label))
        apos.append(np.sum(train_targets == class_label)/train_targets.shape[0])
        ...
    likelihoods = []

    for i in range(test_features.shape[0]):
        likelihood = np.zeros(len(classes))

        for c in range(len(classes)):
            likelihood[c] = apos[c]*likelihood_of_class(test_features[i,:], means[c], covs[c])
        ...
        #print(likelihood)
        likelihoods.append(likelihood)

    return np.array(likelihoods)
    

def test_accuracy():
    acc1, acc2 = [],[]
    for i in range(1):
        pred1 = predict(likelihoods)
        pred2 = predict(maximum_aposteriori(train_features, train_targets, test_features, classes))
        acc1.append(accuracy_score(test_targets, pred1))
        acc2.append(accuracy_score(test_targets, pred2))

    #print(acc1!=acc2)
    #print(np.mean(acc1))
    #print(np.mean(acc2))
    #print(len(acc1))

def change_iris(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    classrm: int,
    rm_ratio: int
):
    #train_targets == class
    n = np.count_nonzero(train_targets == classrm)
    arr = np.zeros(n)

    arr[:np.int(((1-rm_ratio)*n))]  = 1
    np.random.shuffle(arr)
    arr = np.array(arr, dtype=bool)

    noclassindexes = np.where(train_targets != classrm)
    classindexes = np.where(train_targets == classrm)

    rest_features = train_features[noclassindexes]
    rest_targets = train_targets[noclassindexes]

    features = train_features[classindexes]
    targets = train_targets[classindexes]

    red_features = features[arr]
    red_targets = targets[arr]


    return (np.concatenate((rest_features, red_features)),np.concatenate((rest_targets, red_targets)))

    ...

def test_accuracy2():
    red_features, red_targets = change_iris(train_features, train_targets, 2, 0.6)
    #red_features, red_targets = change_iris(red_features, red_targets, 2, 0.2)
    acc1, acc2 = [],[]
    
    plot_points(train_features, train_targets, classes)
    plot_points(red_features, red_targets, classes)

    print(train_features.shape)
    print(red_features.shape)
    #plt.plot(red_features[:,0], red_features[:,1], 'o', color='black')
    #plt.plot(red_features[:,2], red_features[:,3], 'o', color='blue')
    #plt.show

    #print(test_features, len(test_targets))

    for i in range(1):
        pred1 = predict(maximum_likelihood(red_features, red_targets, test_features, classes))
        pred2 = predict(maximum_aposteriori(red_features, red_targets, test_features, classes))
        acc1.append(accuracy_score(test_targets, pred1))
        acc2.append(accuracy_score(test_targets, pred2))

    #print(len(test_targets))

    #print(pred1!=test_targets)
    #print(pred2!=test_targets)
    print(np.mean(acc1))
    print(np.mean(acc2))
    #print(len(acc1))

def plot_points(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list
):  
    plt.clf

    #targets.shape[1]/2
    for n in range(1):
        colors = ['yellow', 'purple', 'blue']
        for i in range(features.shape[0]):
            
            #print(features[i,:2])
            [x, y] = features[i,:(n*2+2)]
            
            #if features[i] == point_targets[i]:
            plt.scatter(x, y, c=colors[targets[i]])
            #else:
            #    plt.scatter(x, y, c=colors[point_targets[i]], edgecolors='red', linewidths=2)

        plt.title('Yellow=0, Purple=1, Blue=2')
        plt.xlabel("Sepal length [cm]")
        plt.ylabel("Sepal width [cm]")

    plt.show()


features, targets, classes = load_iris()
(train_features, train_targets), (test_features, test_targets)\
    = split_train_test(features, targets, train_ratio=0.7)


test_accuracy2()


likelihoods = maximum_likelihood(train_features, train_targets, test_features, classes)
#print(predict(likelihoods))
#mean_of_class(train_features, train_targets, 0)

#covar_of_class(train_features, train_targets, 0)

#class_mean = mean_of_class(train_features, train_targets, 0)
#class_cov = covar_of_class(train_features, train_targets, 0)

#likelihood_of_class(test_features[0, :], class_mean, class_cov)

#maximum_likelihood(train_features, train_targets, test_features, classes)





#print(likelihoods)

#print(predict(maximum_aposteriori(train_features, train_targets, test_features, classes)))

#print(confusion_matrix(pred1, test_targets, classes))
#print(confusion_matrix(pred2, test_targets, classes))
#print(confusion_matrix_my(pred1, test_targets, classes))

#test_accuracy()
#print(pred1[pred1 != test_targets])
#print(test_targets[pred1 != test_targets])