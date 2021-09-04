from typing import Union
import numpy as np
import matplotlib.pyplot as plt
from tools import load_iris, split_train_test
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import pyvista as pv
def sigmoid(x: float) -> float:
    '''
    Calculate the sigmoid of x
    '''
    if x<-100:
        return 0.0
    else:
        return 1/(1+np.exp(-x))
    ...


def d_sigmoid(x: float) -> float:
    '''
    Calculate the derivative of the sigmoid of x.
    '''
    sx=sigmoid(x)
    return sx * (1-sx)
    ...


def perceptron(
    x: np.ndarray,
    w: np.ndarray
) -> Union[float, float]:
    '''
    Return the weighted sum of x and w as well as
    the result of applying the sigmoid activation
    to the weighted sum
    '''
    res = np.dot(x,w)
    return res, sigmoid(res)
    ...


def ffnn(
    x: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray,
) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Computes the output and hidden layer variables for a
    single hidden layer feed-forward neural network.
    '''
    z0 = np.insert(x, 0, 1.0)
    z1 = np.zeros(M)
    a1 = np.zeros(M)
    a2 = np.zeros(K)
    y = np.zeros(K)

    for m in range(M):
        a1[m], z1[m] = perceptron(z0, W1[:, m])
    z1 = np.insert(z1, 0, 1.0)

    for k in range(K):
        a2[k], y[k] = perceptron(z1, W2[:, k])

    


    return y, z0, z1, a1, a2
    ...



def backprop(
    x: np.ndarray,
    target_y: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray
) -> Union[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Perform the backpropagation on given weights W1 and W2
    for the given input pair x, target_y
    '''
    y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)

    dk = y- target_y
    dj = np.zeros(M+1)
    for j in range(M):
        sum = 0.0
        for k in range(K):
            sum += dk[k] * W2[j+1, k]
        dj[j] = d_sigmoid(a1[j]) * sum


    dE1 = np.zeros(W1.shape)
    dE2 = np.zeros(W2.shape)



    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            dE1[i, j] = dj[j]*z0[i]
    
    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            dE2[i, j] = dk[j]*z1[i]

    return y, dE1, dE2

def train_nn(
    X_train: np.ndarray,
    t_train: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray,
    iterations: int,
    eta: float
) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Train a network by:
    1. forward propagating an input feature through the network
    2. Calculate the error between the prediction the network
    made and the actual target
    3. Backpropagating the error through the network to adjust
    the weights.
    '''
    N, D = X_train.shape
    Errtotal = np.zeros(iterations)
    missclass = np.zeros(iterations)

    for iter in range(iterations):
        guess_corr = 0
        Ee = 0
        guesses = np.zeros(N)
        dE1_total = np.zeros(W1.shape)
        dE2_total = np.zeros(W2.shape)

        for n in range(N):
            target_y = np.zeros(K)
            target_y[t_train[n]] = 1
            x = X_train[n, :]

            y, dE1, dE2 = backprop(x, target_y, M, K, W1, W2)

            dE1_total = dE1_total + dE1
            dE2_total = dE2_total + dE2

            if np.argmax(y) == t_train[n]:
                guess_corr += 1
            guesses[n] = np.argmax(y)

            for ik in range(K):
                Ee = Ee - target_y[ik] * np.log(y[ik]) - (1 - target_y[ik]) * np.log(1 - y[ik])

        Errtotal[iter] = Ee / N
        missclass[iter] = 1-guess_corr / N
        W1 = W1 - eta * dE1_total / N
        W2 = W2 - eta * dE2_total / N


    return W1, W2, Errtotal, missclass, guesses

def test_nn(
    X: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray
) -> np.ndarray:
    '''
    Return the predictions made by a network for all features
    in the test set X.
    '''

    N, D = np.shape(X)

    guesses = np.zeros(N)
    for i in range(N):
        x = X[i, :]
        y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)
        guesses[i] = np.argmax(y)
    return guesses


    ...


#print(sigmoid(0.5))
#print(d_sigmoid(0.2))

#print(perceptron(np.array([1.0, 2.3, 1.9]),np.array([0.2,0.3,0.1])))

features, targets, classes = load_iris()
(train_features, train_targets), (test_features, test_targets) = \
    split_train_test(features, targets)
'''
# Take one point:
x = train_features[0, :]
K = 3 # number of classes
M = 10
D=len(train_features[0, :])
# Initialize two random weight matrices
W1 = 2 * np.random.rand(D + 1, M) - 1
W2 = 2 * np.random.rand(M + 1, K) - 1

y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)

print(y)
print(z0)
print(z1)
print(a1)
print(a2)

target_y = [1,0,0]

K = 3  # number of classes
M = 6
D = train_features.shape[1]

x = features[0, :]

# create one-hot target for the feature
target_y = np.zeros(K)
target_y[targets[0]] = 1.0

np.random.seed(42)
# Initialize two random weight matrices
W1 = 2 * np.random.rand(D + 1, M) - 1
W2 = 2 * np.random.rand(M + 1, K) - 1

y, dE1, dE2 = backprop(x, target_y, M, K, W1, W2)

'''



K = 3  # number of classes
M = 6
D = train_features.shape[1]
np.random.seed(42)
# Initialize two random weight matrices
W1 = 2 * np.random.rand(D + 1, M) - 1
W2 = 2 * np.random.rand(M + 1, K) - 1
W1tr, W2tr, Etotal, misclassification_rate, last_guesses = train_nn(
    train_features[:20, :], train_targets[:20], M, K, W1, W2, 500, 0.3)

#print(misclassification_rate)


guesses = test_nn(test_features, M, K, W1tr, W2tr)

#print(guesses)

#print(accuracy_score(test_targets, guesses))

cmt= confusion_matrix(test_targets, guesses)


def plotit():
    errors=[]
    misclass=[]
    iters= []
    for iter in range(1,500,5):
        W1tr, W2tr, Etotal, misclassification_rate, last_guesses = train_nn(train_features[:20, :], train_targets[:20], M, K, W1, W2, iter, 0.3)
        errors.append(Etotal[-1])

        misclass.append(misclassification_rate[-1])
        iters.append(iter)
        print(iter)

    plt.plot(iters,errors, label='Error')    
    plt.plot(iters,misclass, label='Misclassification rate')    
    plt.title('Error and Misclassification rate in relation to training iterations')
    plt.legend(loc='upper right')
    plt.xlabel("Training iterations")
    plt.savefig("./05_backprop/1_1_1.png")
    plt.show()
    ...
#plotit()


import itertools
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("./05_backprop/cmt.png")

#plot_confusion_matrix(cmt, classes)


def testit():
    errors=[]
    misclass=[]
    iters= []
    iter=200
    nodess = []

    acc = []


    for nodes in range(1,20,1):
        print(nodes)
        for iter in range(1,500,10):
            print(iter)

            W1 = 2 * np.random.rand(D + 1, nodes) - 1
            W2 = 2 * np.random.rand(nodes + 1, K) - 1

            W1tr, W2tr, Etotal, misclassification_rate, last_guesses = train_nn(train_features[:20, :], train_targets[:20], nodes, K, W1, W2, iter, 0.3)
            errors.append(Etotal[-1])

            misclass.append(misclassification_rate[-1])
            nodess.append(nodes)
            guesses = test_nn(test_features, nodes, K, W1tr, W2tr)
            iters.append(iter)
            acc.append(accuracy_score(test_targets, guesses))


    #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_trisurf(iters, nodess, acc, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_zlim(0, 0.8)
    plt.show()
    '''
    plt.plot(nodess,acc, label='Accuracy')    
    plt.plot(nodess,errors, label='Errors')    
    plt.plot(nodess,misclass, label='Misclassification rate')    
    plt.title('Accuracy, Error and Misclassification rate in relation to NN Nodes')
    plt.legend(loc='upper right')
    plt.xlabel("Nodes")
    plt.savefig("./05_backprop/Bonus_1.png")
    plt.show()
    '''
#testit()

def testit2():
    errors=[]
    misclass=[]
    iters= []
    iter=200
    nodess = []
    lrs=[]
    acc = []
    nodes =6


    for lr in np.linspace(0.01,1,10):
        print(lr)
        for iter in range(1,500,20):
            print(iter)

            W1 = 2 * np.random.rand(D + 1, nodes) - 1
            W2 = 2 * np.random.rand(nodes + 1, K) - 1

            W1tr, W2tr, Etotal, misclassification_rate, last_guesses = train_nn(train_features[:20, :], train_targets[:20], nodes, K, W1, W2, iter, lr)
            errors.append(Etotal[-1])

            misclass.append(misclassification_rate[-1])
            lrs.append(lr)
            guesses = test_nn(test_features, nodes, K, W1tr, W2tr)
            iters.append(iter)
            acc.append(accuracy_score(test_targets, guesses))


    #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #ax = plt.axes(projection='3d')

    #ax.contour3D(iters, lrs, acc, 50, cmap='binary')    
    #ax.scatter3D(iters, lrs, acc, c=acc, cmap='Greens');
    #plt.show()
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_trisurf(iters, lrs, acc, cmap=cm.coolwarm, linewidth=0.2, antialiased=True)
    ax.zaxis.set_major_locator(LinearLocator(10))
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_zlim(0, 2)
    plt.savefig("./05_backprop/Bonus_3.png")
    plt.show()
    
    '''
    plt.plot(iters,acc, label='Accuracy')    
    plt.plot(iters,lrs, label='Lrs')    
    #plt.plot(nodess,misclass, label='Misclassification rate')    
    plt.title('Accuracy, Error and Misclassification rate in relation to NN Nodes')
    plt.legend(loc='upper right')
    plt.xlabel("Nodes")
    #plt.savefig("./05_backprop/Bonus_1.png")
    plt.show()
    '''
testit2()