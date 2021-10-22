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
from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from itertools import product
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold



def get_even_better_titanic():
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


    Title_Dictionary = {
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Jonkheer": "Royalty",
        "Don": "Royalty",
        "Sir" : "Royalty",
        "Dr": "Officer",
        "Rev": "Officer",
        "the Countess":"Royalty",
        "Mme": "Mrs",
        "Mlle": "Miss",
        "Ms": "Mrs",
        "Mr" : "Mr",
        "Mrs" : "Mrs",
        "Miss" : "Miss",
        "Master" : "Master",
        "Lady" : "Royalty"}

    X_full['Title'] = X_full['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    
    # a map of more aggregated title
    # we map each title
    X_full['Title'] = X_full.Title.map(Title_Dictionary)

    # age
    grouped_train = X_full.iloc[:891].groupby(['Sex','Pclass','Title'])
    grouped_median_train = grouped_train.median()
    grouped_median_train = grouped_median_train.reset_index()[['Sex', 'Pclass', 'Title', 'Age']]

    grouped_median_train.head()

    def fill_age(row):
        condition = (
            (grouped_median_train['Sex'] == row['Sex']) & 
            (grouped_median_train['Title'] == row['Title']) & 
            (grouped_median_train['Pclass'] == row['Pclass'])
        ) 
        return grouped_median_train[condition]['Age'].values[0]

        # a function that fills the missing values of the Age variable
    X_full['Age'] = X_full.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)
    

    #clean the Name variable

    X_full.drop('Name', axis=1, inplace=True)
    
    # encoding in dummy variable
    titles_dummies = pd.get_dummies(X_full['Title'], prefix='Title')
    X_full = pd.concat([X_full, titles_dummies], axis=1)
    
    # removing the title variable
    X_full.drop('Title', axis=1, inplace=True)
    
    
    # add missing fares
    X_full.Fare.fillna(X_full.iloc[:891].Fare.mean(), inplace=True)

    # We drop multiple columns that contain a lot of NaN values
    # in this assignment
    # Maybe we should
    X_full.drop(
        ['PassengerId'],
        inplace=True, axis=1)


    # Instead of dropping the Embarked column we replace NaN values
    # with `S` denoting Southampton, the most common embarking
    # location
    X_full['Embarked'].fillna('S', inplace=True)


    embarked_dummies = pd.get_dummies(X_full['Embarked'], prefix='Embarked')
    X_full = pd.concat([X_full, embarked_dummies], axis=1)
    X_full.drop('Embarked', axis=1, inplace=True)


    X_full.Cabin.fillna('U', inplace=True)
    
    # mapping each Cabin value with the cabin letter
    X_full['Cabin'] = X_full['Cabin'].map(lambda c: c[0])
    
    # dummy encoding ...
    cabin_dummies = pd.get_dummies(X_full['Cabin'], prefix='Cabin')    
    X_full = pd.concat([X_full, cabin_dummies], axis=1)

    X_full.drop('Cabin', axis=1, inplace=True)



    X_full['Sex'] = X_full['Sex'].map({'male':1, 'female':0})
    # We then use the get_dummies function to transform text
    # and non-numerical values into binary categories.
    X_dummies = pd.get_dummies(
        X_full,
        columns=['Cabin_mapped'],
        drop_first=True)


    # encoding into 3 categories:
    pclass_dummies = pd.get_dummies(X_full['Pclass'], prefix="Pclass")
    
    # adding dummy variable
    X_full = pd.concat([X_full, pclass_dummies],axis=1)
    
    # removing "Pclass"
    X_full.drop('Pclass',axis=1,inplace=True)



    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
    def cleanTicket(ticket):
        ticket = ticket.replace('.','')
        ticket = ticket.replace('/','')
        ticket = ticket.split()
        ticket = map(lambda t : t.strip(), ticket)
        ticket = filter(lambda t : not t.isdigit(), ticket)
        ticketlist = list(filter(lambda t : not t.isdigit(), ticket))
        if len(ticketlist) > 0:
            return ticketlist[0]
        else: 
            return 'XXX'
    
    # Extracting dummy variables from tickets:

    X_full['Ticket'] = X_full['Ticket'].map(cleanTicket)
    tickets_dummies = pd.get_dummies(X_full['Ticket'], prefix='Ticket')
    X_full = pd.concat([X_full, tickets_dummies], axis=1)
    X_full.drop('Ticket', inplace=True, axis=1)



    X_full['FamilySize'] = X_full['Parch'] + X_full['SibSp'] + 1
    
    # introducing other features based on the family size
    X_full['Singleton'] = X_full['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    X_full['SmallFamily'] = X_full['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
    X_full['LargeFamily'] = X_full['FamilySize'].map(lambda s: 1 if 5 <= s else 0)


    # We now have the cleaned data we can use in the assignment
    X = X_full[:len(train)]
    submission_X = X_full[len(train):]
    y = train.Survived
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.3, random_state=5, stratify=y)

    return (X_train, y_train), (X_test, y_test), submission_X


def train_test_pred(c, X_train, t_train, X_test, t_test, submission_X):

    c.fit(X_train, t_train)

    y_pred = c.predict(X_test)
    y_subpred = c.predict(submission_X)

    acc = accuracy_score(t_test, y_pred)
    prec = precision_score(t_test, y_pred)
    rec = recall_score(t_test, y_pred)

    return acc, prec, rec, y_subpred


def param_search_GB(X, y):
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
        verbose=2,
        n_iter=500,
        cv=4,
        n_jobs=-1)
    # Fit randomized_mse to the data
    gb_random.fit(X, y)
    # Print the best parameters and lowest RMSE
    return gb_random.best_params_

def param_search_KNN(X, y):
    '''
    Perform randomized parameter search on the
    gradient boosting classifier on the dataset (X, y)
    '''
    # Create the parameter grid
    param_grid = {
        'n_neighbors': sp_randInt(1,20), 
        'weights': ['uniform', 'distance'], 
        'metric': ['euclidean', 'manhattan']}
    # Instantiate the regressor
    c = KNeighborsClassifier()
    # Perform random search
    random = RandomizedSearchCV(
        param_distributions=param_grid,
        estimator=c,
        scoring="accuracy",
        verbose=0,
        n_iter=500,
        cv=4,
        n_jobs=-1)
    # Fit randomized_mse to the data
    results = random.fit(X, y)
    # Print the best parameters and lowest RMSE
    return results.best_params_, results.best_score_

def param_search_RF(X, y):
    '''
    Perform randomized parameter search on the
    gradient boosting classifier on the dataset (X, y)
    '''

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    param_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

    # Instantiate the regressor
    c = RandomForestClassifier()
    # Perform random search
    random = RandomizedSearchCV(
        param_distributions=param_grid,
        estimator=c,
        scoring="accuracy",
        verbose=2,
        n_iter=100,
        cv=4,
        n_jobs=-1)
    # Fit randomized_mse to the data
    results = random.fit(X, y)
    # Print the best parameters and lowest RMSE
    return results.best_params_, results.best_score_

def param_search_ADA(X, y):
    '''
    Perform randomized parameter search on the
    gradient boosting classifier on the dataset (X, y)
    '''
    # Create the parameter grid
    param_grid = {
        'n_estimators': sp_randInt(1,500),
        'learning_rate': sp_randFloat()}
    # Instantiate the regressor
    c = AdaBoostClassifier()
    # Perform random search
    random = RandomizedSearchCV(
        param_distributions=param_grid,
        estimator=c,
        scoring="accuracy",
        verbose=2,
        n_iter=100,
        cv=4,
        n_jobs=-1)
    # Fit randomized_mse to the data
    results = random.fit(X, y)
    # Print the best parameters and lowest RMSE
    return results.best_params_, results.best_score_

def param_search_SVM(X, y):
    '''
    Perform randomized parameter search on the
    gradient boosting classifier on the dataset (X, y)
    '''
    # Create the parameter grid
    param_grid = {
        'C': sp_randFloat(1,500),
        'gamma': sp_randFloat(0.01,2),
        'kernel': ['poly']}
    # Instantiate the regressor
    c = SVC()
    # Perform random search
    random = RandomizedSearchCV(
        param_distributions=param_grid,
        estimator=c,
        scoring="accuracy",
        verbose=2,
        n_iter=5,
        cv=4,
        n_jobs=-1)
    # Fit randomized_mse to the data
    results = random.fit(X, y)
    # Print the best parameters and lowest RMSE
    return results.best_params_, results.best_score_

def param_search_DT(X, y):
    '''
    Perform randomized parameter search on the
    gradient boosting classifier on the dataset (X, y)
    '''
    # Create the parameter grid
    param_grid = {
        "max_depth": [3, None],
        "max_features": sp_randInt(1, 9),
        "min_samples_leaf": sp_randInt(1, 9),
        "criterion": ["gini", "entropy"]}
    # Instantiate the regressor
    c = DecisionTreeClassifier()
    # Perform random search
    random = RandomizedSearchCV(
        param_distributions=param_grid,
        estimator=c,
        scoring="accuracy",
        verbose=2,
        n_iter=1000,
        cv=4,
        n_jobs=-1)
    # Fit randomized_mse to the data
    results = random.fit(X, y)
    # Print the best parameters and lowest RMSE
    return results.best_params_, results.best_score_

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

def param_search_extra(X, y):
    '''
    Perform randomized parameter search on the
    gradient boosting classifier on the dataset (X, y)
    '''
    # Create the parameter grid
    bla = range(200,500,10)
    blub = range(1,10,2)
    lr = linspace(0.05,0.5,10)
    mins = range(2, 15, 3)
    minsl = range(2, 15, 3)
    max_features = range(1,32,2)

    et_param_grid = {
        'n_estimators': bla,
        'max_depth': blub,
        #'learning_rate': lr,
        'min_samples_split': mins,
        'min_samples_leaf': minsl,
        #'max_features': max_features
        #'n_estimators': sp_randInt(1,500),
        'max_features': max_features,
        #'learning_rate': sp_randFloat()
        #range(100,400,50)
        }
    # Instantiate the regressor
    et = ExtraTreesClassifier()
    # Perform random search
    et_random = RandomizedSearchCV(
        param_distributions=et_param_grid,
        estimator=et,
        scoring="f1",
        verbose=0,
        n_iter=500,
        cv=4,
        n_jobs=-1)
    # Fit randomized_mse to the data
    et_random.fit(X, y)
    # Print the best parameters and lowest RMSE
    return et_random.best_params_





def _create_submission(res):
    '''Create your kaggle submission
    '''
    c = GradientBoostingClassifier(n_estimators=res["n_estimators"], max_depth=res["max_depth"], learning_rate=res["learning_rate"])
    c.fit(tr_X, tr_y)
    prediction = c.predict(submission_X)
    build_kaggle_submission(prediction)




(tr_X, tr_y), (tst_X, tst_y), submission_X = get_even_better_titanic()



'''
cs=[RandomForestClassifier(),
    GradientBoostingClassifier(),
    KNeighborsClassifier(),
    svm.SVC(kernel='rbf'),
    svm.SVC(kernel='poly'),
    svm.SVC(kernel='linear')
    ]

y_preds = np.empty((len(submission_X),len(cs)))
acc = np.empty(len(cs))

for i in range(len(cs)):
    res = train_test_pred(cs[i], tr_X, tr_y, tst_X, tst_y, submission_X)

    y_preds[:,i] = res[3]

print(y_preds[1,:])

'''








clf = RandomForestClassifier(n_estimators=500, max_depth = 5)
clf = clf.fit(tr_X, tr_y)



features = pd.DataFrame()
features['feature'] = tr_X.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)

features.plot(kind='barh', figsize=(25, 25))

#plt.show()


model = SelectFromModel(clf, prefit=True)
train_reduced = model.transform(tr_X)
#print(train_reduced.shape)
# (891L, 14L)

test_reduced = model.transform(tst_X)
#print(test_reduced.shape)

sub_reduced = model.transform(submission_X)


#print(param_search_KNN(tr_X, tr_y))
#print(param_search_RF(tr_X, tr_y))
#print(param_search_ADA(tr_X, tr_y))
#print(param_search_SVM(tr_X, tr_y))
#print(param_search_DT(tr_X, tr_y))
#print(param_search_GB(tr_X, tr_y))

#print(param_search_KNN(train_reduced, tr_y))
#print(param_search_RF(train_reduced, tr_y))
#print(param_search_ADA(train_reduced, tr_y))
#print(param_search_SVM(train_reduced, tr_y))
#print(param_search_DT(train_reduced, tr_y))
#print(param_search_GB(train_reduced, tr_y))


'''

clf1 = DecisionTreeClassifier(criterion = 'gini', max_depth = None, max_features = 5, min_samples_leaf = 4)
clf2 = KNeighborsClassifier(metric= 'manhattan', n_neighbors= 12, weights= 'distance')
clf3 = svm.SVC(C = 336.65633965721327, gamma = 0.010393303078475516, kernel = 'rbf', probability = True )
clf4 = GradientBoostingClassifier(learning_rate = 0.20666807393051, max_depth = 4, n_estimators = 16, subsample = 0.5852835816723386)
clf5 = RandomForestClassifier(n_estimators = 200, min_samples_split = 2, min_samples_leaf = 4, max_features = 'auto', max_depth = 50, bootstrap = True)
clf8 = AdaBoostClassifier(learning_rate = 0.0737370583165543, n_estimators = 276)

clf4 = GradientBoostingClassifier()
clf5 = RandomForestClassifier()
clf8 = AdaBoostClassifier()

#eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3), ('GB', clf4), ('rf', clf5), ('ada', clf8)], voting='hard')

eclf = VotingClassifier(estimators=[ ('GB', clf4), ('rf', clf5), ('ada', clf8)], voting='hard')


eclf.fit(tr_X, tr_y)

y_pred = eclf.predict(tst_X)
y_predsub = eclf.predict(submission_X)
acc = accuracy_score(tst_y, y_pred)
print(np.mean(cross_val_score(eclf, tr_X, tr_y)))



eclf.fit(train_reduced, tr_y)

y_pred = eclf.predict(test_reduced)
y_predsub = eclf.predict(sub_reduced)
acc = accuracy_score(tst_y, y_pred)
print(np.mean(cross_val_score(eclf, train_reduced, tr_y)))

build_kaggle_submission(y_predsub)




for clf, label in zip([clf1, clf2, clf3, clf4, clf5, clf6, clf7, clf8, eclf], ['dt', 'knn', 'svcrbf', 'gb', 'rf', 'svcl', 'svcp', 'ada', 'Ensemble']):
    scores = cross_val_score(clf, tr_X, tr_y, scoring='accuracy', cv=5)
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))





def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)
    return np.mean(xval)

logreg = LogisticRegression()
logreg_cv = LogisticRegressionCV()
rf = RandomForestClassifier()
gboost = GradientBoostingClassifier()

models = [logreg, logreg_cv, rf, gboost]

for model in models:
    print('Cross-validation of : {0}'.format(model.__class__))
    score = compute_score(clf=model, X=train_reduced, y=tr_y, scoring='accuracy')
    print('CV score = {0}'.format(score))
    print('****')



run_gs = False

if run_gs:
    parameter_grid = {
                 'max_depth' : [4, 6, 8],
                 'n_estimators': [50, 10],
                 'max_features': ['sqrt', 'auto', 'log2'],
                 'min_samples_split': [2, 3, 10],
                 'min_samples_leaf': [1, 3, 10],
                 'bootstrap': [True, False],
                 }
    forest = RandomForestClassifier()
    cross_validation = StratifiedKFold(n_splits=5)

    grid_search = GridSearchCV(forest,
                               scoring='accuracy',
                               param_grid=parameter_grid,
                               cv=cross_validation,
                               verbose=1
                              )

    grid_search.fit(tr_X, tr_y)
    model = grid_search
    parameters = grid_search.best_params_

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))
    
else: 
    parameters = {'bootstrap': True, 'max_depth': 8, 'max_features': 'auto', 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 10}
    
    model = RandomForestClassifier(**parameters)
    model.fit(tr_X, tr_y)

    y_pred = model.predict(tst_X)
    y_predsub = model.predict(submission_X)
    acc = accuracy_score(tst_y, y_pred)
    print(np.mean(cross_val_score(model, tr_X, tr_y)))


res = param_search(tr_X, tr_y)
print(res)

res = {'n_estimators': 88, 'max_depth': 8, 'learning_rate': 0.09090909090909091}
c = GradientBoostingClassifier(**res)
c.fit(tr_X, tr_y)
y_pred = c.predict(tst_X)

acc = accuracy_score(tst_y, y_pred)
print(np.mean(cross_val_score(c, tr_X, tr_y)))

'''
res = param_search_extra(tr_X, tr_y)
print(res)


clf = ExtraTreesClassifier(**res)
clf.fit(tr_X, tr_y)
y=clf.predict(tst_X)

y_sub=clf.predict(submission_X)
acc = accuracy_score(tst_y, y)
print(acc)
print(np.mean(cross_val_score(clf, tr_X, tr_y)))
build_kaggle_submission(y_sub)