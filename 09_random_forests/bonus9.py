import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (confusion_matrix, accuracy_score, recall_score,
                             precision_score)

from collections import OrderedDict
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.metrics import make_scorer


class WineClassifier:
    '''
    A general class to try out different sklearn classifiers
    on the wine dataset
    '''
    def __init__(self, classifier, train_ratio: float = 0.7):
        self.classifier = classifier
        wine = load_wine()
        self.X = wine.data  # all feature vectors
        self.header = wine.feature_names
        self.t = wine.target  # all corresponding labels
        self.X_train, self.X_test, self.t_train, self.t_test =\
            train_test_split(
                wine.data, wine.target,
                test_size=1-train_ratio, random_state=109)

        # Fit the classifier to the training data here
        #print(self.X.shape)
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
        return precision_score(self.t_test, self.pred, average=None)

    def recall(self) -> float:
        '''Returns the recall on the test data
        '''
        return recall_score(self.t_test, self.pred, average=None)

    
        
    
    def cross_validation_accuracy(self) -> float:
        '''Returns the average 10-fold cross validation
        accuracy on the entire dataset.
        '''
        return cross_val_score(self.classifier, self.X, self.t, cv=4).mean()

    def cross_validation_cm(self) -> float:
        '''Returns the average 10-fold cross validation
        accuracy on the entire dataset.
        '''
        y_pred = cross_val_predict(self.classifier, self.X, self.t, cv=4)
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



def find_largest():
    res = []

    for e in range(50,400,50):
        
        for f in range(1,5):
            print(e,f)    
            classifier_type = RandomForestClassifier(n_estimators=e, max_features=f)
            cc = WineClassifier(classifier_type)
            res.append([e,f,cc.cross_validation_accuracy()])

    res=np.array(res)
    index = np.argsort(res[:,2],)
    #print(res[index[::-1]][:3])

    return res[index[::-1]][:1]

def find_largest_extra():
    res = []

    for e in range(100,400,50):
        
        for f in range(5,15,2):
            print(e,f)    
            classifier_type = ExtraTreesClassifier(n_estimators=e, max_depth=f,warm_start=True)
            cc = WineClassifier(classifier_type)
            res.append([e,f,cc.cross_validation_accuracy()])

    res=np.array(res)
    index = np.argsort(res[:,2],)
    #print(res[index[::-1]][:3])

    return res[index[::-1]][:1]



'''
classifier_type = DecisionTreeClassifier()
cc = wineClassifier(classifier_type)

print("CM: ",cc.confusion_matrix())
print("Accuracy: ",np.round(cc.accuracy(),4))
print("Precision: ",np.round(cc.precision(),4))
print("Recall: ",np.round(cc.recall(),4))
print("Cross validation: ",np.round(cc.cross_validation_accuracy(),4))
print("Confmat cross: ",cc.cross_validation_cm())
'''

#bla = find_largest()

#print(bla)





classifier_type = RandomForestClassifier()
cc = WineClassifier(classifier_type)




print("CM: ",cc.confusion_matrix())
print("Accuracy: ",np.round(cc.accuracy(),4))
print("Cross validation: ",np.round(cc.cross_validation_accuracy(),4))
print("Confmat cross: ",cc.cross_validation_cm())



classifier_type = RandomForestClassifier(n_estimators=350, max_features=1)
cc = WineClassifier(classifier_type)

print("CM: ",cc.confusion_matrix())
print("Accuracy: ",np.round(cc.accuracy(),4))
print("Cross validation: ",np.round(cc.cross_validation_accuracy(),4))
print("Confmat cross: ",cc.cross_validation_cm())

'''
#_plot_oob_error()


classifier_type = ExtraTreesClassifier()
cc = WineClassifier(classifier_type)

print("CM: ",cc.confusion_matrix())
print("Accuracy: ",np.round(cc.accuracy(),4))
print("Cross validation: ",np.round(cc.cross_validation_accuracy(),4))
print("Confmat cross: ",cc.cross_validation_cm())


classifier_type = ExtraTreesClassifier(n_estimators=200, max_depth=8)
cc = WineClassifier(classifier_type)

print("CM: ",cc.confusion_matrix())
print("Accuracy: ",np.round(cc.accuracy(),4))
print("Cross validation: ",np.round(cc.cross_validation_accuracy(),4))
print("Confmat cross: ",cc.cross_validation_cm())




#_plot_extreme_oob_error()
'''