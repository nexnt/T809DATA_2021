import pandas as pd
import numpy as np
import re
# visualization
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import (train_test_split, RandomizedSearchCV)
from sklearn.metrics import (confusion_matrix, accuracy_score, recall_score,
                             precision_score)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

                             
def get_better_titanic():
    '''
    Loads the cleaned titanic dataset but change
    how we handle the age column.
    Loads the cleaned titanic dataset
    '''

    # Load in the raw data
    # check if data directory exists for Mimir submissions
    # DO NOT REMOVE

    train =  pd.read_csv('./12_PCA/data/train.csv')
    test = pd.read_csv('./12_PCA/data/test.csv')



    # Concatenate the train and test set into a single dataframe
    # we drop the `Survived` column from the train set
    X_full = pd.concat([train.drop('Survived', axis=1), test], axis=0)
    

    X_full = X_full.drop(['PassengerId','Name','Cabin','Ticket'],axis=1)



    X_full['Fare']=X_full.Fare.fillna(X_full.Fare.mean())



    X_full.loc[X_full['Age']<= 18, 'Age'] = 0
    X_full.loc[(X_full['Age']> 18) & (X_full['Age']<= 32), 'Age'] =1 
    X_full.loc[(X_full['Age']> 32) & (X_full['Age']<=48), 'Age'] = 2
    X_full.loc[(X_full['Age']> 48) & (X_full['Age']<=64), 'Age'] = 3
    X_full.loc[X_full['Age']> 64, 'Age'] = 4

    


    X_full.Embarked.value_counts()
    Fre_embarked_package = X_full.Embarked.mode()
    Fre_age_band = X_full.Age.mode()
    
    X_full['Age']=X_full.Age.fillna(Fre_age_band[0])
    X_full['Embarked']=X_full.Embarked.fillna(Fre_embarked_package[0])

    X_full=pd.get_dummies(X_full,columns=['Sex','Embarked'],drop_first=True)
    
    print(X_full.corr())

    df_1 = X_full.loc[:,['Fare','Pclass']]

    pca =  PCA(n_components=1)
    col_1 = pca.fit_transform(df_1)


    X_full['Mod_col_1']=col_1[:,0]


    X_full=X_full.drop(['Fare','Pclass'], axis=1)
 


    df_3 = X_full.loc[:,['SibSp','Parch']]

    pca =  PCA(n_components=1)
    col_3 = pca.fit_transform(df_3)


    X_full['Mod_col_2']=col_3[:,0]

    X_full=X_full.drop(['SibSp','Parch'], axis=1)

    
    
    # We now have the cleaned data we can use in the assignment
    X = X_full[:len(train)]
    submission_X = X_full[len(train):]
    y = train.Survived
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.3, random_state=5, stratify=y)

    #print(y)
    return (X_train, y_train), (X_test, y_test), submission_X



def plot_pca():


    titanic = pd.read_csv('./12_PCA/data/train.csv')
    titanic_test = pd.read_csv('./12_PCA/data/test.csv')
    test = titanic_test

    Male_titanic = titanic.loc[titanic.Sex== 'male',:]
    Female_titanic = titanic.loc[titanic.Sex== 'female',:]

    l0=sns.FacetGrid(Male_titanic, col='Survived')
    l0.map(plt.hist, 'Age', bins=20)
    plt.savefig("./12_PCA/Bonus_Male_1.png")
    l1=sns.FacetGrid(Male_titanic, col='Survived')
    l1.map(plt.hist, 'Fare', bins=20)
    plt.savefig("./12_PCA/Bonus_Male_2.png")
    l2=sns.FacetGrid(Male_titanic, col='Survived', row='Pclass')
    l2.map(plt.hist, 'Age', bins=20)
    plt.savefig("./12_PCA/Bonus_Male_3.png")
    l3=sns.FacetGrid(Male_titanic, col='Survived')
    l3.map(plt.hist, 'Age', bins=20)
    plt.savefig("./12_PCA/Bonus_Male_4.png")
    l4=sns.FacetGrid(Male_titanic, col='Survived', row='Embarked')
    l4.map(plt.hist, 'Age', bins=20)
    plt.savefig("./12_PCA/Bonus_Male_5.png")

    l0=sns.FacetGrid(Female_titanic, col='Survived')
    l0.map(plt.hist, 'Age', bins=20)
    plt.savefig("./12_PCA/Bonus_Female_1.png")
    l1=sns.FacetGrid(Female_titanic, col='Survived')
    l1.map(plt.hist, 'Fare', bins=20)
    plt.savefig("./12_PCA/Bonus_Female_2.png")
    l2=sns.FacetGrid(Female_titanic, col='Survived', row='Pclass')
    l2.map(plt.hist, 'Age', bins=20)
    plt.savefig("./12_PCA/Bonus_Female_3.png")
    l3=sns.FacetGrid(Female_titanic, col='Survived')
    l3.map(plt.hist, 'Age', bins=20)
    plt.savefig("./12_PCA/Bonus_Female_4.png")
    l4=sns.FacetGrid(Female_titanic, col='Survived', row='Embarked')
    l4.map(plt.hist, 'Age', bins=20)
    plt.savefig("./12_PCA/Bonus_Female_5.png")

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
    max_features = range(1,X.shape[1],2)

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


def build_kaggle_submission(prediction):
    '''
    Given a prediction, this function will build a
    kaggle compatible submission and save it to disk
    at ./data/your_submission.csv`.
    '''
    test = pd.read_csv('./12_PCA/data/test.csv')
    submission = pd.concat(
        [test.PassengerId, pd.DataFrame(prediction)],
        axis='columns')
    submission.columns = ["PassengerId", "Survived"]
    submission.to_csv('./12_PCA/data/your_submission.csv', header=True, index=False)
    print('Your submission can be found at ./12_PCA/data/your_submission.csv')

(tr_X, tr_y), (tst_X, tst_y), submission_X = get_better_titanic()


#res = param_search_extra(tr_X, tr_y)
#print(res)
res = {'n_estimators': 200, 'min_samples_split': 11, 'min_samples_leaf': 2, 'max_features': 6, 
'max_depth': 4}
print(tr_X.shape[1])
clf = ExtraTreesClassifier(**res)
clf.fit(tr_X, tr_y)
y=clf.predict(tst_X)

y_sub=clf.predict(submission_X)
acc = accuracy_score(tst_y, y)
print(acc)
print(np.mean(cross_val_score(clf, tr_X, tr_y)))
build_kaggle_submission(clf.predict(submission_X))

