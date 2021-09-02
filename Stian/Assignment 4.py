from tools import load_iris, split_train_test

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


def mean_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the mean of a selected class given all features
    and targets in a dataset
    '''
    return np.mean(
        features[targets == selected_class, :], axis=0)


def covar_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the covariance of a selected class given all
    features and targets in a dataset
    '''
    return np.cov(
        features[targets == selected_class, :], rowvar=False)


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
    return multivariate_normal(
        mean=class_mean, cov=class_covar).pdf(feature)


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
    means = []
    covs = []
    for class_label in classes:
        means.append(mean_of_class(
            train_features,
            train_targets,
            class_label))
        covs.append(covar_of_class(
            train_features,
            train_targets,
            class_label))

    likelihoods = []
    for i in range(test_features.shape[0]):
        likelihoods.append([
            likelihood_of_class(
                test_features[i, :], z[0], z[1]) for z in zip(means, covs)
        ])
    return np.array(likelihoods)


def predict(likelihoods: np.ndarray):
    '''
    Given an array of shape [num_datapoints x num_classes]
    make a prediction for each datapoint by choosing the
    highest likelihood.

    You should return a [likelhoods.shape[0]] shaped numpy
    array of predictions, e.g. [0, 1, 0, ..., 1, 2]
    '''
    guesses = np.argmax(np.array(likelihoods), axis=1)
    return guesses


def maximum_aposteriori(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:
    means = []
    covs = []
    priors = []
    for class_label in classes:
        means.append(mean_of_class(
            train_features,
            train_targets,
            class_label))
        covs.append(covar_of_class(
            train_features,
            train_targets,
            class_label))
        priors.append(np.sum(
            train_targets == class_label)/train_targets.shape[0])

    likelihoods = []
    for i in range(test_features.shape[0]):
        likelihoods.append([
            z[2]*likelihood_of_class(test_features[i, :], z[0], z[1])
            for z in zip(means, covs, priors)
        ])
    return np.array(likelihoods)


features, targets, classes = load_iris()
(train_features, train_targets), (test_features, test_targets)\
    = split_train_test(features, targets, train_ratio=0.6)

class_mean = mean_of_class(train_features, train_targets, 0)
class_cov = covar_of_class(train_features, train_targets, 0)
likelihood_of_class(test_features[0, :], class_mean, class_cov)






# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(train_features)
X_test = sc.transform(train_features)

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(train_features, train_targets)

# Predicting the Test set results
y_pred = classifier.predict(test_features)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
ac = accuracy_score(test_targets,y_pred)
cm = confusion_matrix(test_targets, y_pred)

print(ac, cm)