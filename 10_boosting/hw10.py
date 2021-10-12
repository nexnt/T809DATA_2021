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

    X_full['Age'].fillna(X_full.Age.mean(), inplace=True)


    # We then use the get_dummies function to transform text
    # and non-numerical values into binary categories.
    X_dummies = pd.get_dummies(
        X_full,
        columns=['Sex', 'Cabin_mapped', 'Embarked'],
        drop_first=True)

    # We now have the cleaned data we can use in the assignment
    X = X_dummies[:len(train)]
    submission_X = X_dummies[len(train):]
    y = train.Survived
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.3, random_state=5, stratify=y)

    print(y)
    return (X_train, y_train), (X_test, y_test), submission_X

def rfc_train_test(X_train, t_train, X_test, t_test):
    '''
    Train a random forest classifier on (X_train, t_train)
    and evaluate it on (X_test, t_test)
    '''

    c = RandomForestClassifier()
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
    gb_param_grid = {
        'n_estimators': sp_randInt(1,500),
        'max_depth': sp_randInt(4, 10),
        'learning_rate': sp_randFloat(),
        "subsample"    : sp_randFloat()}
    # Instantiate the regressor
    gb = GradientBoostingClassifier()
    # Perform random search
    gb_random = RandomizedSearchCV(
        param_distributions=gb_param_grid,
        estimator=gb,
        scoring="accuracy",
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
    c = GradientBoostingClassifier(n_estimators=res["n_estimators"], max_depth=res["max_depth"], learning_rate=res["learning_rate"])
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



(tr_X, tr_y), (tst_X, tst_y), submission_X = get_better_titanic()




acc, prec, rec = rfc_train_test(tr_X, tr_y, tst_X, tst_y)

print("Accuracy: ",np.round(acc,4))
print("Precision: ",np.round(prec,4))
print("Recall: ",np.round(rec,4))
'''
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