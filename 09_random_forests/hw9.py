import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (confusion_matrix, accuracy_score, recall_score,
                             precision_score)

from collections import OrderedDict
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.metrics import make_scorer


class CancerClassifier:
    '''
    A general class to try out different sklearn classifiers
    on the cancer dataset
    '''
    def __init__(self, classifier, train_ratio: float = 0.7):
        self.classifier = classifier
        cancer = load_breast_cancer()
        self.X = cancer.data  # all feature vectors
        self.header = cancer.feature_names
        self.t = cancer.target  # all corresponding labels
        self.X_train, self.X_test, self.t_train, self.t_test =\
            train_test_split(
                cancer.data, cancer.target,
                test_size=1-train_ratio, random_state=109)

        # Fit the classifier to the training data here
        
        self.classifier.fit(self.X_train, self.t_train)
        self.pred = self.classifier.predict(self.X_test)

    def confusion_matrix(self) -> np.ndarray:
        '''Returns the confusion matrix on the test data
        '''
        return confusion_matrix(self.t_test, self.pred)

    def accuracy(self) -> float:
        '''Returns the accuracy on the test data
        '''
        return accuracy_score(self.t_test, self.pred)

    def precision(self) -> float:
        '''Returns the precision on the test data
        '''
        return precision_score(self.t_test, self.pred)

    def recall(self) -> float:
        '''Returns the recall on the test data
        '''
        return recall_score(self.t_test, self.pred)

    
        
    
    def cross_validation_accuracy(self) -> float:
        '''Returns the average 10-fold cross validation
        accuracy on the entire dataset.
        '''
        return cross_val_score(self.classifier, self.X, self.t, cv=10).mean()

    def cross_validation_cm(self) -> float:
        '''Returns the average 10-fold cross validation
        accuracy on the entire dataset.
        '''
        y_pred = cross_val_predict(self.classifier, self.X, self.t, cv=10)
        conf_mat = confusion_matrix(self.t, y_pred)
        return conf_mat

    def feature_importance(self) -> list:
        '''
        Draw and show a barplot of feature importances
        for the current classifier and return a list of
        indices, sorted by feature importance (high to low).
        '''
        feature_importance = self.classifier.feature_importances_
        print(feature_importance)
        index = np.argsort(feature_importance)[::-1]
        print(feature_importance[index])
        plt.bar(np.array(range(0,len(feature_importance))),feature_importance)
        plt.xlabel("Feature index")
        plt.ylabel("Feature importance")
        plt.title("Feature importances for the current classifier")
        plt.savefig("09_random_forests/3_1_1.png")
        #plt.show()
        print(self.header[index])
        return index

def _plot_oob_error():
    RANDOM_STATE = 1337
    ensemble_clfs = [
        ("RandomForestClassifier, max_features='sqrt'",
            RandomForestClassifier(
                n_estimators=100,
                warm_start=True,
                oob_score=True,
                max_features="sqrt",
                random_state=RANDOM_STATE)),
        ("RandomForestClassifier, max_features='log2'",
            RandomForestClassifier(
                n_estimators=100,
                warm_start=True,
                max_features='log2',
                oob_score=True,
                random_state=RANDOM_STATE)),
        ("RandomForestClassifier, max_features=None",
            RandomForestClassifier(
                n_estimators=100,
                warm_start=True,
                max_features=None,
                oob_score=True,
                random_state=RANDOM_STATE))]

    # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

    min_estimators = 30
    max_estimators = 175
    classifier_type = RandomForestClassifier()
    cc = CancerClassifier(classifier_type)

    for label, clf in ensemble_clfs:
        for i in range(min_estimators, max_estimators + 1):
            clf.set_params(n_estimators=i)
            clf.fit(cc.X, cc.t)  # Use cancer data here
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))

    # Generate the "OOB error rate" vs. "n_estimators" plot.
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.savefig("09_random_forests/2_4_1.png")
    plt.show()


def _plot_extreme_oob_error():
    RANDOM_STATE = 1337
    ensemble_clfs = [
        ("ExtraTreesClassifier, max_features='sqrt'",
            ExtraTreesClassifier(
                n_estimators=100,
                warm_start=True,
                oob_score=True,
                max_features="sqrt",
                bootstrap=True,
                random_state=RANDOM_STATE)),
        ("ExtraTreesClassifier, max_features='log2'",
            ExtraTreesClassifier(
                n_estimators=100,
                warm_start=True,
                max_features='log2',
                oob_score=True,
                bootstrap=True,
                random_state=RANDOM_STATE)),
        ("ExtraTreesClassifier, max_features=None",
            ExtraTreesClassifier(
                n_estimators=100,
                warm_start=True,
                max_features=None,
                oob_score=True,
                bootstrap=True,
                random_state=RANDOM_STATE))]

    # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

    min_estimators = 30
    max_estimators = 175
    classifier_type = ExtraTreesClassifier()
    cc = CancerClassifier(classifier_type)

    for label, clf in ensemble_clfs:
        for i in range(min_estimators, max_estimators + 1):
            clf.set_params(n_estimators=i)
            clf.fit(cc.X, cc.t)  # Use cancer data here
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))

    # Generate the "OOB error rate" vs. "n_estimators" plot.
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.savefig("09_random_forests/3_2_1.png")
    plt.show()
    
    ...


'''
classifier_type = DecisionTreeClassifier()
cc = CancerClassifier(classifier_type)

print("CM: ",cc.confusion_matrix())
print("Accuracy: ",np.round(cc.accuracy(),4))
print("Precision: ",np.round(cc.precision(),4))
print("Recall: ",np.round(cc.recall(),4))
print("Cross validation: ",np.round(cc.cross_validation_accuracy(),4))
print("Confmat cross: ",cc.cross_validation_cm())
'''


def find_largest():
    res = []

    for e in range(100,400,50):
        
        for f in range(1,30,5):
            print(e,f)    
            classifier_type = RandomForestClassifier(n_estimators=e, max_features=f)
            cc = CancerClassifier(classifier_type)
            res.append([e,f,cc.accuracy(),cc.precision(),cc.recall(),cc.cross_validation_accuracy()])

    res=np.array(res)
    index = np.argsort(res[:,2],)
    #print(res[index[::-1]][:3])

    return res[index[::-1]][:1]



bla = find_largest()

print(bla)



classifier_type = RandomForestClassifier()
cc = CancerClassifier(classifier_type)




print("CM: ",cc.confusion_matrix())
print("Accuracy: ",np.round(cc.accuracy(),4))
print("Precision: ",np.round(cc.precision(),4))
print("Recall: ",np.round(cc.recall(),4))
print("Cross validation: ",np.round(cc.cross_validation_accuracy(),4))
print("Confmat cross: ",cc.cross_validation_cm())




classifier_type = RandomForestClassifier(n_estimators=371, max_features=9)
cc = CancerClassifier(classifier_type)

print("CM: ",cc.confusion_matrix())
print("Accuracy: ",np.round(cc.accuracy(),4))
print("Precision: ",np.round(cc.precision(),4))
print("Recall: ",np.round(cc.recall(),4))
print("Cross validation: ",np.round(cc.cross_validation_accuracy(),4))
print("Confmat cross: ",cc.cross_validation_cm())


'''

classifier_type = RandomForestClassifier(n_estimators=9, max_features=13)
cc = CancerClassifier(classifier_type)

feature_idx = cc.feature_importance()

print(feature_idx)

#_plot_oob_error()

classifier_type = ExtraTreesClassifier()
cc = CancerClassifier(classifier_type)
feature_idx = cc.feature_importance()

print(feature_idx)

print("CM: ",cc.confusion_matrix())
print("Accuracy: ",np.round(cc.accuracy(),4))
print("Precision: ",np.round(cc.precision(),4))
print("Recall: ",np.round(cc.recall(),4))
print("Cross validation: ",np.round(cc.cross_validation_accuracy(),4))
print("Confmat cross: ",cc.cross_validation_cm())


#_plot_extreme_oob_error()

'''