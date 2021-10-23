import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import cumsum
from sklearn.decomposition import PCA

from tools import load_cancer


def standardize(X: np.ndarray) -> np.ndarray:
    '''
    Standardize an array of shape [N x 1]

    Input arguments:
    * X (np.ndarray): An array of shape [N x 1]

    Returns:
    (np.ndarray): A standardized version of X, also
    of shape [N x 1]
    '''
    ...
    return (X - np.mean(X, axis=0)) / np.std(X,axis=0)




def scatter_standardized_dims(
    X: np.ndarray,
    i: int,
    j: int,
):
    '''
    Plots a scatter plot of N points where the n-th point
    has the coordinate (X_ni, X_nj)

    Input arguments:
    * X (np.ndarray): A [N x f] array
    * i (int): The first index
    * j (int): The second index
    '''
    stdX = standardize(X)
    plt.scatter(stdX[:,i],stdX[:,j])


def _scatter_cancer():
    X, y, head = load_cancer()

    for i in range(X.shape[1]):

        plt.subplot(5, 6, i+1)
        plt.title("{} vs {}".format(head[0], head[i]))

        scatter_standardized_dims(X, 0, i)
    
    plt.subplots_adjust(left=0.1, 
                    bottom=0.1,  
                    right=0.9,  
                    top=0.9,  
                    wspace=0.4,  
                    hspace=0.4) 
    plt.plot()

def _plot_pca_components():
    ...
    X, y, head = load_cancer()
    stdX = standardize(X)
    pca=PCA(n_components=X.shape[1])
    pca.fit_transform(stdX)
    print(pca.components_)
    for i in range(X.shape[1]):
        plt.subplot(5, 6, i+1)
        plt.title("PCA {}".format(i+1))
        plt.plot(pca.components_[i])
    plt.subplots_adjust(left=0.1, 
                    bottom=0.1,  
                    right=0.9,  
                    top=0.9,  
                    wspace=0.4,  
                    hspace=0.4) 
    plt.show()


def _plot_eigen_values():
    X, y, head = load_cancer()
    stdX = standardize(X)
    plt.clf()
    pca=PCA(n_components=X.shape[1])
    pca.fit_transform(stdX)

    plt.plot(pca.explained_variance_)
    
    plt.xlabel('Eigenvalue index')
    plt.ylabel('Eigenvalue')
    plt.grid()
    plt.savefig("./12_PCA/3_1_1.png")
    plt.show()



def _plot_log_eigen_values():
    X, y, head = load_cancer()
    stdX = standardize(X)
    plt.clf()
    pca=PCA(n_components=X.shape[1])
    pca.fit_transform(stdX)

    plt.plot(np.log10(pca.explained_variance_))
    plt.xlabel('Eigenvalue index')
    plt.ylabel('$\log_{10}$ Eigenvalue')
    plt.grid()
    plt.savefig("./12_PCA/3_2_1.png")
    plt.show()


def _plot_cum_variance():
    X, y, head = load_cancer()
    stdX = standardize(X)
    plt.clf()
    pca=PCA(n_components=X.shape[1])
    pca.fit_transform(stdX)
    I=pca.explained_variance_.shape[0]
    plt.plot([np.cumsum(pca.explained_variance_)[i]/sum(pca.explained_variance_) for i in range(pca.explained_variance_.shape[0])])
    plt.xlabel('Eigenvalue index')
    plt.ylabel('Percentage variance')
    plt.grid()
    plt.savefig("./12_PCA/3_3_1.png")
    plt.show()


X, y, head = load_cancer()


#_scatter_cancer()
#plt.show()
#plt.savefig("./12_PCA/1_3_1.png")

#_plot_pca_components()
#_plot_eigen_values()
#_plot_log_eigen_values()

_plot_cum_variance()