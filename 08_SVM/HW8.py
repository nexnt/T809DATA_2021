from matplotlib.colors import LinearSegmentedColormap
from numpy.core.fromnumeric import mean
import sklearn
from tools import plot_svm_margin, load_cancer
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score)
from matplotlib.offsetbox import AnchoredText
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import timeit

def _plot_linear_kernel():
    X, t = make_blobs(n_samples=40, centers=2)
    clf = svm.SVC(kernel='linear',C=1000)
    clf.fit(X,t)
    plot_svm_margin(clf, X, t)
    #plt.show()
    #plt.savefig("08_SVM/1_1_1.png")


def _subplot_svm_margin(
    svc,
    X: np.ndarray,
    t: np.ndarray,
    num_plots: int,
    index: int
):
    '''
    Plots the decision boundary and decision margins
    for a dataset of features X and labels t and a support
    vector machine svc.

    Input arguments:
    * svc: An instance of sklearn.svm.SVC: a C-support Vector
    classification model
    * X: [N x f] array of features
    * t: [N] array of target labels
    '''
    # similar to tools.plot_svm_margin but added num_plots and
    # index where num_plots should be the total number of plots
    # and index is the index of the current plot being generated
    
    plt.subplot(1,num_plots,index)

    plt.scatter(X[:, 0], X[:, 1], c=t, s=30, cmap=plt.cm.Paired)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    Z = svc.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(
        XX, YY, Z,
        colors='k', levels=[-1, 0, 1],
        alpha=0.5, linestyles=['--', '-', '--'])

    # plot support vectors
    ax.scatter(
        svc.support_vectors_[:, 0],
        svc.support_vectors_[:, 1],
        s=100, linewidth=1, facecolors='none', edgecolors='k')

    anchored_text = AnchoredText("N SV\nC1: {}\nC2: {}\nShp: {}".format(svc.n_support_[0],svc.n_support_[1], svc.decision_function_shape) , loc=2)
    ax.add_artist(anchored_text)





def _compare_gamma():
    X, t = make_blobs(n_samples=40, centers=2, random_state=6)

    clf = svm.SVC(kernel='rbf',C=1000)
    clf.fit(X,t)
    
    _subplot_svm_margin(clf, X, t, 3, 1)
    anchored_text = AnchoredText("Gamma: {}".format(clf.gamma) , loc=3)
    ax = plt.gca()
    ax.add_artist(anchored_text)


    clf = svm.SVC(kernel='rbf',C=1000, gamma=0.2)
    clf.fit(X,t)
    ...
    _subplot_svm_margin(clf, X, t, 3, 2)
    anchored_text = AnchoredText("Gamma: {}".format(clf.gamma) , loc=3)
    ax = plt.gca()
    ax.add_artist(anchored_text)

    clf = svm.SVC(kernel='rbf',C=1000, gamma=2)
    clf.fit(X,t)

    _subplot_svm_margin(clf, X, t, 3, 3)
    anchored_text = AnchoredText("Gamma: {}".format(clf.gamma) , loc=3)
    ax = plt.gca()
    ax.add_artist(anchored_text)
    plt.savefig("08_SVM/1_3_1.png")
    plt.show()


def _compare_C():
    X, t = make_blobs(n_samples=40, centers=2, random_state=6)
    plt.figure(figsize=(15,10)) 
    clf = svm.SVC(kernel='linear',C=1000)
    clf.fit(X,t)

    _subplot_svm_margin(clf, X, t, 5, 1)
    anchored_text = AnchoredText("C: {}".format(clf.C) , loc=3)
    ax = plt.gca()
    ax.add_artist(anchored_text)
    clf = svm.SVC(kernel='linear',C=0.5)
    clf.fit(X,t)

    _subplot_svm_margin(clf, X, t, 5, 2)
    anchored_text = AnchoredText("C: {}".format(clf.C) , loc=3)
    ax = plt.gca()
    ax.add_artist(anchored_text)
    clf = svm.SVC(kernel='linear',C=0.3)
    clf.fit(X,t)
    

    _subplot_svm_margin(clf, X, t, 5, 3)
    anchored_text = AnchoredText("C: {}".format(clf.C) , loc=3)
    ax = plt.gca()
    ax.add_artist(anchored_text)
    clf = svm.SVC(kernel='linear',C=0.05)
    clf.fit(X,t)

    _subplot_svm_margin(clf, X, t, 5, 4)
    anchored_text = AnchoredText("C: {}".format(clf.C) , loc=3)
    ax = plt.gca()
    ax.add_artist(anchored_text)
    clf = svm.SVC(kernel='linear',C=0.0001)
    clf.fit(X,t)

    _subplot_svm_margin(clf, X, t, 5, 5)
    anchored_text = AnchoredText("C: {}".format(clf.C) , loc=3)
    ax = plt.gca()
    ax.add_artist(anchored_text)
    plt.savefig("08_SVM/1_5_1.png")
    plt.show()


def train_test_SVM(
    svc,
    X_train: np.ndarray,
    t_train: np.ndarray,
    X_test: np.ndarray,
    t_test: np.ndarray,
):
    '''
    Train a configured SVM on <X_train> and <t_train>
    and then measure accuracy, precision and recall on
    the test set

    This function should return (accuracy, precision, recall)
    '''
    start = timeit.default_timer()
    svc.fit(X_train,t_train)
    
    y = svc.predict(X_test)
    end = timeit.default_timer()
    print(end-start)
    return accuracy_score(t_test,y), precision_score(t_test, y), recall_score(t_test, y), (end-start)

#_plot_linear_kernel()
#_compare_gamma()
#_compare_C()



res=np.empty((3,4))

fcns=['linear', 'rbf', 'poly']



for i in range(3):
    accs, precs, recs, times = [], [], [], []
    for n in range(20):    

        (X_train, t_train), (X_test, t_test) = load_cancer()
        svc = svm.SVC(kernel=fcns[i],C=1000)
        acc, prec, rec, time = train_test_SVM(svc, X_train, t_train, X_test, t_test)
        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        times.append(time)
        print(i,n)
    res[i] = [mean(accs), mean(precs), mean(recs), mean(times)] 

a=('acc',
'prec',
'rec',
'time')

fcns=(['linear'], 
    ['rbf'], 
    ['poly'])

tab = np.append(np.array(fcns), res, axis=1)

print (tabulate(tab, headers=a))

