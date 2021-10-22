import os
import numpy as np
from numpy.core.function_base import linspace
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split,
                                     RandomizedSearchCV)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, recall_score,
                             precision_score)

from tools import get_titanic, build_kaggle_submission
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt

def get_better_titanic():
    '''
    Loads the cleaned titanic dataset but change
    how we handle the age column.
    Loads the cleaned titanic dataset
    '''

    # Load in the raw data
    # check if data directory exists for Mimir submissions
    # DO NOT REMOVE
    if os.path.exists('./10_boosting/data/train.csv'):
        train = pd.read_csv('./10_boosting/data/train.csv')
        test = pd.read_csv('./10_boosting/data/test.csv')
    else:
        train = pd.read_csv('train.csv')
        test = pd.read_csv('test.csv')

    # Concatenate the train and test set into a single dataframe
    # we drop the `Survived` column from the train set
    X_full = pd.concat([train.drop('Survived', axis=1), test], axis=0)
    
    # The cabin category consist of a letter and a number.
    # We can divide the cabin category by extracting the first
    # letter and use that to create a new category. So before we
    # drop the `Cabin` column we extract these values
    X_full['Cabin_mapped'] = X_full['Cabin'].astype(str).str[0]
    # Then we transform the letters into numbers
    cabin_dict = {k: i for i, k in enumerate(X_full.Cabin_mapped.unique())}
    X_full.loc[:, 'Cabin_mapped'] =\
        X_full.loc[:, 'Cabin_mapped'].map(cabin_dict)

    # We drop multiple columns that contain a lot of NaN values
    # in this assignment
    # Maybe we should
    X_full.drop(
        ['PassengerId', 'Cabin', 'Name', 'Ticket'],
        inplace=True, axis=1)

    # Instead of dropping the fare column we replace NaN values
    # with the 3rd class passenger fare mean.
    fare_mean = X_full[X_full.Pclass == 3].Fare.mean()
    X_full['Fare'].fillna(fare_mean, inplace=True)
    # Instead of dropping the Embarked column we replace NaN values
    # with `S` denoting Southampton, the most common embarking
    # location
    X_full['Embarked'].fillna('S', inplace=True)

    







    # We then use the get_dummies function to transform text
    # and non-numerical values into binary categories.
    X_dummies = pd.get_dummies(
        X_full,
        columns=['Sex', 'Cabin_mapped', 'Embarked'],
        drop_first=True)


    
    X_dummies['Age'].fillna(X_dummies.Age.mean(), inplace=True)
    '''
    X_Age = X_dummies[X_dummies.Age.notna()].drop(['Age'], axis=1)
    y_Age = X_dummies[X_dummies.Age.notna()]['Age']
    X_pred = X_dummies[X_dummies.Age.isna()].drop(['Age'], axis=1)

    print(X_dummies[X_dummies.Age.notna()].drop(['Age'], axis=1))
    from sklearn import svm
    c = svm.SVR(kernel='rbf')
    c.fit(X_Age,y_Age)
    #X_dummies[X_dummies['Age'].isna()]['Age'] = c.predict(X_pred)

    X_dummies.loc[X_dummies.Age.isna(),'Age'] = c.predict(X_pred)
    '''
    # We now have the cleaned data we can use in the assignment
    X = X_dummies[:len(train)]
    submission_X = X_dummies[len(train):]
    y = train.Survived
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.3, random_state=5, stratify=y)

    #print(y)
    return (X_train, y_train), (X_test, y_test), submission_X

def rfc_train_test(X_train, t_train, X_test, t_test):
    '''
    Train a random forest classifier on (X_train, t_train)
    and evaluate it on (X_test, t_test)
    '''

    c = RandomForestClassifier(max_features='log2')
    c.fit(X_train, t_train)
    y_pred = c.predict(X_test)

    acc = accuracy_score(t_test, y_pred)
    prec = precision_score(t_test, y_pred)
    rec = recall_score(t_test, y_pred)

    return acc, prec, rec

def gb_train_test(X_train, t_train, X_test, t_test):
    '''
    Train a Gradient boosting classifier on (X_train, t_train)
    and evaluate it on (X_test, t_test)
    '''
    c = GradientBoostingClassifier()
    c.fit(X_train, t_train)
    y_pred = c.predict(X_test)

    acc = accuracy_score(t_test, y_pred)
    prec = precision_score(t_test, y_pred)
    rec = recall_score(t_test, y_pred)

    return acc, prec, rec

def param_search(X, y):
    '''
    Perform randomized parameter search on the
    gradient boosting classifier on the dataset (X, y)
    '''
    # Create the parameter grid
    bla = range(1,100,1)
    blub = range(1,32,1)
    lr = linspace(0.05,0.5,100)
    mins = np.linspace(0.1, 1.0, 10, endpoint=True)
    minsl = np.linspace(0.1, 0.5, 5, endpoint=True)
    max_features = range(1,32,32)

    gb_param_grid = {
        'n_estimators': bla,
        'max_depth': blub,
        'learning_rate': lr,
        #'min_samples_split': mins,
        #'min_samples_leaf': minsl,
        #'max_features': max_features
        #'n_estimators': sp_randInt(1,500),
        #'max_depth': sp_randInt(2, 10),
        #'learning_rate': sp_randFloat()
        #range(100,400,50)
        }
    # Instantiate the regressor
    gb = GradientBoostingClassifier()
    # Perform random search
    gb_random = RandomizedSearchCV(
        param_distributions=gb_param_grid,
        estimator=gb,
        scoring="f1",
        verbose=0,
        n_iter=500,
        cv=4,
        n_jobs=-1)
    # Fit randomized_mse to the data
    gb_random.fit(X, y)
    # Print the best parameters and lowest RMSE
    return gb_random.best_params_


def gb_optimized_train_test(X_train, t_train, X_test, t_test, res):
    '''
    Train a gradient boosting classifier on (X_train, t_train)
    and evaluate it on (X_test, t_test) with
    your own optimized parameters
    '''
    #c = GradientBoostingClassifier(n_estimators=res["n_estimators"], max_depth=res["max_depth"], learning_rate=res["learning_rate"], min_samples_split=res["min_samples_split"], min_samples_leaf=res["min_samples_leaf"], max_features=res["max_features"])
    c = GradientBoostingClassifier(n_estimators=68, max_depth=4, learning_rate=0.2)
    c.fit(X_train, t_train)
    y_pred = c.predict(X_test)

    acc = accuracy_score(t_test, y_pred)
    prec = precision_score(t_test, y_pred)
    rec = recall_score(t_test, y_pred)

    return acc, prec, rec


def _create_submission(res):
    '''Create your kaggle submission
    '''
    c = GradientBoostingClassifier(n_estimators=res["n_estimators"], max_depth=res["max_depth"], learning_rate=res["learning_rate"])
    c.fit(tr_X, tr_y)
    prediction = c.predict(submission_X)
    build_kaggle_submission(prediction)



(tr_X, tr_y), (tst_X, tst_y), submission_X = get_titanic()

res = {'n_estimators': 87, 'max_depth': 7, 'learning_rate': 0.48636363636363633}

_create_submission(res)

'''
acc, prec, rec = rfc_train_test(tr_X, tr_y, tst_X, tst_y)

print("Accuracy: ",np.round(acc,4))
print("Precision: ",np.round(prec,4))
print("Recall: ",np.round(rec,4))


acc, prec, rec = gb_train_test(tr_X, tr_y, tst_X, tst_y)

print("Accuracy: ",np.round(acc,4))
print("Precision: ",np.round(prec,4))
print("Recall: ",np.round(rec,4))


res = param_search(tr_X, tr_y)
print(res)

acc, prec, rec = gb_optimized_train_test(tr_X, tr_y, tst_X, tst_y, res)

print("Accuracy: ",np.round(acc,4))
print("Precision: ",np.round(prec,4))
print("Recall: ",np.round(rec,4))

_create_submission(res)

res["acc"] = acc

with open("./10_boosting/sample.txt", "a") as file_object:
    # Append 'hello' at the end of file
    
    file_object.write(str(res))
    file_object.write("\n")
'''