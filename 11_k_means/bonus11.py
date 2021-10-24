from math import nan
import numpy as np
import sklearn as sk
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from typing import Union
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
from sklearn.preprocessing import StandardScaler
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

from tools import load_iris, image_to_numpy, plot_gmm_results
import seaborn as sns

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
    '''
    X_PCA = X
    X_PCA['Survived']= y

    X_PCA.corr().to_csv("corr.csv")
    cor = X_PCA.corr()
    print(cor)
    cor_target = abs(cor["Survived"])
    relevant_features = cor_target[cor_target>0.5]
    print(relevant_features)
    '''
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.3, random_state=5, stratify=y)

    return (X_train, y_train), (X_test, y_test), submission_X

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

def get_titanic():
    '''
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
        ['PassengerId', 'Cabin', 'Age', 'Name', 'Ticket'],
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

    # We now have the cleaned data we can use in the assignment
    X = X_dummies[:len(train)]
    submission_X = X_dummies[len(train):]
    y = train.Survived
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.3, random_state=5, stratify=y)

    return (X_train, y_train), (X_test, y_test), submission_X

def build_kaggle_submission(prediction):
    '''
    Given a prediction, this function will build a
    kaggle compatible submission and save it to disk
    at ./data/your_submission.csv`.
    '''
    test = pd.read_csv('./10_boosting/data/test.csv')
    submission = pd.concat(
        [test.PassengerId, pd.DataFrame(prediction)],
        axis='columns')
    submission.columns = ["PassengerId", "Survived"]
    submission.to_csv('./11_k_means/your_submission.csv', header=True, index=False)
    print('Your submission can be found at ./11_k_means/your_submission.csv')


def distance_matrix(
    X: np.ndarray,
    Mu: np.ndarray
) -> np.ndarray:
    '''
    Returns a matrix of euclidian distances between points in
    X and Mu.

    Input arguments:
    * X (np.ndarray): A [n x f] array of samples
    * Mu (np.ndarray): A [k x f] array of prototypes

    Returns:
    out (np.ndarray): A [n x k] array of euclidian distances
    where out[i, j] is the euclidian distance between X[i, :]
    and Mu[j, :]
    '''
    n,f = X.shape
    k,f = Mu.shape
    distances = np.zeros((n,k))
    for i in range(n):
        for j in range(k):
            distances[i,j] = np.sqrt(sum(np.power(X[i,:] - Mu[j,:],2)))
    return distances


def determine_r(dist: np.ndarray) -> np.ndarray:
    '''
    Returns a matrix of binary indicators, determining
    assignment of samples to prototypes.

    Input arguments:
    * dist (np.ndarray): A [n x k] array of distances

    Returns:
    out (np.ndarray): A [n x k] array where out[i, j] is
    1 if sample i is closest to prototype j and 0 otherwise.
    '''
    R = np.zeros((dist.shape))
    for i in range(dist.shape[0]):
        R[i,np.argmin(dist[i,:])] = 1
    return R


def determine_j(R: np.ndarray, dist: np.ndarray) -> float:
    '''
    Calculates the value of the objective function given
    arrays of indicators and distances.

    Input arguments:
    * R (np.ndarray): A [n x k] array where out[i, j] is
        1 if sample i is closest to prototype j and 0
        otherwise.
    * dist (np.ndarray): A [n x k] array of distances

    Returns:
    * out (float): The value of the objective function
    '''
    J = np.zeros((dist.shape))
    for n in range(dist.shape[0]):
        for k in range(dist.shape[1]):
            J[n, k] = R[n, k] * dist[n, k]

    res = sum(sum(J))/J.shape[0]
    return res


def update_Mu(
    Mu: np.ndarray,
    X: np.ndarray,
    R: np.ndarray
) -> np.ndarray:
    '''
    Updates the prototypes, given arrays of current
    prototypes, samples and indicators.

    Input arguments:
    Mu (np.ndarray): A [k x f] array of current prototypes.
    X (np.ndarray): A [n x f] array of samples.
    R (np.ndarray): A [n x k] array of indicators.

    Returns:
    out (np.ndarray): A [k x f] array of updated prototypes.
    '''

    N, K = R.shape
    newmu = np.zeros_like(Mu)
    for k in range(K):
        res1=0
        res2=0
        for n in range(N):
            res1 += R[n,k]*X[n]
            res2 += R[n,k]
        if res2 != 0:
            newmu[k] = res1/res2
        else:
            newmu[k] = Mu[k]
    return newmu
    
    ...


def k_means(
    X_standard: np.ndarray,
    k: int,
    num_its: int
) -> Union[list, np.ndarray, np.ndarray]:
    # We first have to standardize the samples

    # run the k_means algorithm on X_st, not X.

    # we pick K random samples from X as prototypes
    nn = sk.utils.shuffle(range(X_standard.shape[0]))
    Mu = X_standard[nn[0: k], :]

    Js = []

    for i in range(num_its):
        distances = distance_matrix(X_standard, Mu)
        R = determine_r(distances)

        J = determine_j(R, distances)
        Js.append(J)
        Mu = update_Mu(Mu, X_standard, R)


    # Then we have to "de-standardize" the prototypes
    '''
    for i in range(k):
        Mu[i, :] = Mu[i, :] * np.sqrt(ss.var_) + ss.mean_
    '''
    return Mu, R, Js


def _plot_j():

    Mu, R, Js = k_means(X, 4, 10)
    plt.clf
    plt.plot(Js)
    plt.title("Evolution of J")
    plt.ylabel("Value of J")
    plt.xlabel("Iteration")
    plt.savefig("11_k_means/1_6_1.png")
    plt.show()


def _plot_multi_j():
    K= [2,3,5,10]
    plt.clf 
    for k in K:
        Mu, R, Js = k_means(tr_X.to_numpy(), k, 50)
        print(Js)
        plt.plot(Js, label='k = {}'.format(k))
    plt.title("Evolution of J")    
    plt.ylabel("Value of J")
    plt.xlabel("Iteration")
    plt.legend(loc='upper center')
    


    plt.show()

def k_means_predict(
    X: np.ndarray,
    t: np.ndarray,
    classes: list,
    num_its: int,
    k:int
) -> np.ndarray:
    '''
    Determine the accuracy and confusion matrix
    of predictions made by k_means on a dataset
    [X, t] where we assume the most common cluster
    for a class label corresponds to that label.

    Input arguments:
    * X (np.ndarray): A [n x f] array of input features
    * t (np.ndarray): A [n] array of target labels
    * classes (list): A list of possible target labels
    * num_its (int): Number of k_means iterations

    Returns:
    * the predictions (list)
    '''

    ss = StandardScaler()
    Xstd =ss.fit_transform(X)
    Xtststd = ss.transform(tst_X)
    Xsubstd = ss.transform(submission_X)

    nc = len(classes)
    nt = len(t)
    Mu, R, Js = k_means(Xstd, k=k, num_its=num_its)

    R = pd.DataFrame(np.argmax(R, axis=1), columns = ['Cluster'])
    R['Survived'] = t

    survival_mapper = R.groupby('Cluster')['Survived'].mean().round().astype(np.int).to_dict()
    R['group_survived'] = R.Cluster.map(survival_mapper)
    survival_map = R.groupby('Cluster')['Survived'].mean().round().astype(np.int).to_dict()

    accuracy = (R['Survived'] == R['group_survived']).mean()
    
    # predict acc for sub
    distances = distance_matrix(Xsubstd, Mu)
    R_sub = determine_r(distances)
    
    predict = pd.DataFrame(np.argmax(R_sub, axis=1), columns = ['Cluster'])
    test = pd.read_csv('./10_boosting/data/test.csv')
    predict['Survived'] = predict.Cluster.map(survival_map)
    predict['PassengerId'] = test['PassengerId']
    predict = predict[['PassengerId', 'Survived']].sort_values('PassengerId').reset_index(drop=True)
    ### predict tst acc
    distances = distance_matrix(Xtststd, Mu)
    R_tst = determine_r(distances)

    R_tst = pd.DataFrame(np.argmax(R_tst, axis=1), columns = ['Cluster'])
    R_tst['Survived'] = tst_y

    survival_mapper = R_tst.groupby('Cluster')['Survived'].mean().round().astype(np.int).to_dict()
    R_tst['group_survived'] = R_tst.Cluster.map(survival_mapper)
    survival_map = R_tst.groupby('Cluster')['Survived'].mean().round().astype(np.int).to_dict()

    accuracy = (R_tst['Survived'] == R_tst['group_survived']).mean()

    return R, predict, accuracy, Js


def kmeans_sk(
    X: np.ndarray,
    t: np.ndarray,
    classes: list,
    num_its: int,
    k:int
) -> np.ndarray:

    ss = StandardScaler()
    X_std = ss.fit_transform(tr_X)
    Xsubs = ss.fit_transform(submission_X)
    X_tststd = ss.fit_transform(tst_X)
    c = KMeans(n_clusters=k, max_iter=num_its)
    c.fit(X_std, tst_y)
    clst = c.predict(X_tststd)


    R = pd.DataFrame(clst, columns = ['Cluster'])
    R['Survived'] = tr_y
    
    survival_mapper = R.groupby('Cluster')['Survived'].mean().round().astype(np.int).to_dict()
    R['group_survived'] = R.Cluster.map(survival_mapper)
    accuracy = (R['Survived'] == R['group_survived']).mean()
    print(accuracy)

(tr_X, tr_y), (tst_X, tst_y), submission_X = get_better_titanic()



#_plot_multi_j()

tst_y=tst_y.to_numpy()

#pred, predict = k_means_predict(tr_X, tr_y, [0,1], 20, 50)
#pred, predict, accuracy = k_means_predict(tr_X.to_numpy(), tr_y.to_numpy(), [0,1], 20, 100)
#print(accuracy)

accuracys = []
ks = []
Jss =  []
for k in range(10,300,10):
    pred, predict, accuracy, Js = k_means_predict(tr_X.to_numpy(), tr_y.to_numpy(), [0,1], 10, k)
    accuracys.append(accuracy)
    ks.append(k)
    Jss.append(Js)
    print(accuracy)

plt.plot(ks,accuracys)


plt.ylabel("Accuracy on testdata")
plt.xlabel("Number of Clusters")
plt.savefig("11_k_means/Bonus_1.png")
plt.show()
plt.clf
for i in range(len(Jss)):

    plt.plot(range(0,len(Jss[i])),Jss[i], label='k = {}'.format(ks[i]))


plt.ylabel("Value of J")
plt.xlabel("Iteration")
plt.legend(loc='upper right')
plt.savefig("11_k_means/Bonus_2.png")
plt.show()

'''
build_kaggle_submission(predict['Survived'])

accuracy = (pred['Survived'] == pred['group_survived']).mean()
print(accuracy)

pred = kmeans_sk(tr_X, tr_y, [0,1], 150, 47)
'''