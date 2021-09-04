from numpy.core.fromnumeric import shape
from numpy.core.function_base import linspace
from tools import scatter_3d_data, bar_per_axis

import matplotlib.pyplot as plt
import numpy as np


def gen_data(
    n: int,
    k: int,
    mean: np.ndarray,
    var: float
) -> np.ndarray:
    '''Generate n values samples from the k-variate
    normal distribution
    '''

    covar = np.power(var, 2)*np.identity(k, dtype=float)



    #print(np.random.multivariate_normal(mean, covar, n))
    return np.random.multivariate_normal(mean, covar, n)
    '''
    for i in range(result.shape[0]):
        for m in range(result.shape[1]):

            print(np.random.multivariate_normal(np.zeros(3), np.eye(3), size=5))
            print(np.random.multivariate_normal(mean, var ,1))
            #result[i,k] = np.random.multivariate_normal(mean, var)
    '''
    
    #print(result)


def update_sequence_mean(
    mu: np.ndarray,
    x: np.ndarray,
    n: int
) -> np.ndarray:
    '''Performs the mean sequence estimation update
    '''
    #print(mu + (x-mu)/n)
    return mu + (x-mu)/n
    ...


def _plot_sequence_estimate():
    data = gen_data(100, 3, np.array([0, 1, -1]), np.sqrt(3))
    estimates = [np.array([0, 0, 0])]
    for i in range(data.shape[0]):
        estimates.append(update_sequence_mean(estimates[i], data[i, :], i+1))
    plt.clf()
    plt.plot([e[0] for e in estimates], label='First dimension')
    plt.plot([e[1] for e in estimates], label='Second dimension')
    plt.plot([e[2] for e in estimates], label='Third dimension')
    plt.legend(loc='upper center')
    plt.title('Plot sequence estimate Mean: 0, 1, -1 Sigma: Sqrt(3)')
    plt.xlabel("Estimate based on n datapoints")
    plt.ylabel("Mean")
    plt.savefig("./03_sequential_estimation/1_5_1.png")
    plt.show()


def _square_error(y, y_hat):

    return np.power(np.subtract(y, y_hat), 2)


def _plot_mean_square_error():
    origmu = np.array([0, 1, -1])
    data = gen_data(100, 3, origmu, np.sqrt(3))
    estimates = [np.array([0, 0, 0])]
    sqarederror = []
    for i in range(data.shape[0]):
        estimates.append(update_sequence_mean(estimates[i], data[i, :], i+1))
        sqarederror.append(np.mean(_square_error(origmu, estimates[-1])))
        
    
    plt.clf()
    plt.plot(sqarederror, label='Mean of n=3 dimentions')
    plt.legend(loc='upper center')
    plt.title('Squared error plot Mean: 0, 1, -1 Sigma: Sqrt(3)')
    plt.xlabel("Squared error after processing n datapoints")
    plt.ylabel("Sqared Error")
    plt.savefig("./03_sequential_estimation/1_6_1.png")
    plt.show()


# Naive solution to the independent question.

def gen_changing_data(
    n: int,
    k: int,
    start_mean: np.ndarray,
    end_mean: np.ndarray,
    var: float
) -> np.ndarray:
    # remove this if you don't go for the independent section

    ranges = linspace(start_mean, end_mean, 500)
    dataset= np.zeros([n, k])

    for i in range(ranges.shape[0]):
        
        covar = np.power(var, 2)*np.identity(k, dtype=float)
        
        #print(np.random.multivariate_normal(mean, covar, n))
        dataset[i,:] = np.random.multivariate_normal(ranges[i,:], covar, 1)

    
        ...

    return (np.array(dataset), ranges)

def _plot_changing_sequence_estimate():
    # remove this if you don't go for the independent section

    data, ranges = gen_changing_data(500, 3, np.array([0, 1, -1]), np.array([1,-1,0]), np.sqrt(3))

    z = 100
    estimates = [np.array([0, 0, 0])]
    sqarederror = []
    added = np.zeros([500, 3])
    u=0
    for i in range(data.shape[0]):
        

        if i <= z:

            estimates.append(estimates[i]+(data[i, :]-estimates[i] )/(i+1))
            sqarederror.append(np.mean(_square_error(ranges[i,:], estimates[-1])))
            ...
        else: 
            u=u+1
            estimates.append(estimates[i] + (data[i, :]-estimates[i])/z - (data[i-z, :]-estimates[i])/z)
            sqarederror.append(np.mean(_square_error(ranges[i,:], estimates[-1])))
    estimates = np.array(estimates)
    #print(estimates)
    plt.clf()
    plt.plot([e[0] for e in estimates], label='First dimension')
    plt.plot([e[1] for e in estimates], label='Second dimension')
    plt.plot([e[2] for e in estimates], label='Third dimension')
    plt.legend(loc='upper center')
    plt.title('Plot sequence estimate Mean: 0, 1, -1 to 1, -1, 0  Sigma: Sqrt(3)')
    plt.xlabel("Estimate based on range of 50 datapoints")
    plt.ylabel("Mean")
    plt.savefig("./03_sequential_estimation/bonus_1.png")
    plt.show()

    plt.clf()
    plt.plot(sqarederror, label='Mean of n=3 dimentions')
    plt.legend(loc='upper center')
    plt.title('Squared error plot Mean: 0, 1, -1 to 1, -1, 0 Sigma: Sqrt(3)')
    plt.xlabel("Sqared error based on rangen of 50 datapoints")
    plt.ylabel("Sqared Error")
    plt.savefig("./03_sequential_estimation/bonus_2.png")
    plt.show()
    




#gen_changing_data(500, 3, np.array([0, 1, -1]), np.array([1,-1,0]), np.sqrt(3))
#_plot_changing_sequence_estimate()


#gen_data(2, 3, np.array([0, 1, -1]), 1.3)

#gen_data(5, 1, np.array([0.5]), 0.5)

points = gen_data(300, 3, np.array([0, 1, -1]), np.sqrt(3))
#scatter_3d_data(points)
#bar_per_axis(points)

X = points

mean = np.mean(X, 0)
new_x = gen_data(1, 3, np.array([0, 0, 0]), 1)

#print(new_x)
#print(mean)
#update_sequence_mean(mean, new_x, X.shape[0])
#_plot_sequence_estimate()

#_plot_mean_square_error()

mean = np.mean(X, 0)
new_x = gen_data(1, 3, np.array([0, 0, 0]), 1)
print(update_sequence_mean(mean, new_x, X.shape[0]))