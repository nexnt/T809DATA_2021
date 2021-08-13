import numpy as np
import matplotlib.pyplot as plt


def normal(x: np.ndarray, sigma: float, mu: float) -> np.ndarray:
    # Part 1.1
    
    p=1/np.sqrt(2 * np.pi * np.power(sigma, 2)) * np.exp(-np.power(x-mu, 2)/(2 * np.power(sigma, 2)))
    return p
    
def plot_normal(sigma: float, mu:float, x_start: float, x_end: float):
    # Part 1.2

    x_range = np.linspace(x_start, x_end, 500)

    
    plt.plot(x_range, 1/np.sqrt(2 * np.pi * np.power(sigma, 2)) * np.exp(-np.power(x_range-mu, 2)/(2 * np.power(sigma, 2))), label=f'Power ')



def _plot_three_normals():
    # Part 1.2
    plt.clf()

    plot_normal(0.5, 0, -5, 5)
    plot_normal(0.25, 1, -5, 5)
    plot_normal(1, 1.5, -5, 5)

    plt.savefig("./00_introduction/1_2_1.png")


def normal_mixture(x: np.ndarray, sigmas: list, mus: list, weights: list):
    # Part 2.1
    
    result=np.zeros(len(x))

    for i in range(len(sigmas)):
        
        result += weights[i]/np.sqrt(2 * np.pi * np.power(sigmas[i], 2)) * np.exp(-np.power(x-mus[i], 2)/(2 * np.power(sigmas[i], 2)))
    print(result)
    return result
    
        
def _compare_components_and_mixture():
    # Part 2.2
    plt.clf()
    sigmas: list = [0.5, 1.5, 0.25]
    mus: list = [0,-0.5, 1.5]
    weights: list = [1/3, 1/3, 1/3]
    x_range = np.linspace(-5, 5, 500)

    plot_normal(0.5, 0, -5, 5)
    plot_normal(1.5, -0.5, -5, 5)
    plot_normal(0.25, 1.5, -5, 5)

    
    
    result=np.zeros(len(x_range))

    for i in range(len(sigmas)):
        
        result += weights[i]/np.sqrt(2 * np.pi * np.power(sigmas[i], 2)) * np.exp(-np.power(x_range-mus[i], 2)/(2 * np.power(sigmas[i], 2)))

    plt.plot(x_range,result)
    plt.savefig("./00_introduction/2_2_1.png")
    plt.show()

def sample_gaussian_mixture(sigmas: list, mus: list, weights: list, n_samples: int = 500):
# Part 3.1

    n=np.random.multinomial(n_samples, weights)


    result: np.ndarray = []
    

    for i in range(len(sigmas)):
        res=np.random.normal(mus[i], sigmas[i], n[i])
        
        result = np.append(result,res,)
    #print(result)
    return result

  
def _plot_mixture_and_samples():
    # Part 3.2
    n_samples_tot: list = [10, 100, 500, 1000]
    subplot: int = 141
    plt.clf()
    sigmas: list = [0.3, 0.5, 1]
    mus: list = [0, -1, 1.5]
    weights: list = [0.2, 0.3, 0.5]
    x_range = np.linspace(-5, 5, 500)    


    result=np.zeros(len(x_range))

    for i in range(len(sigmas)):
        
        result += weights[i]/np.sqrt(2 * np.pi * np.power(sigmas[i], 2)) * np.exp(-np.power(x_range-mus[i], 2)/(2 * np.power(sigmas[i], 2)))


    for i in range(len(n_samples_tot)):

        plt.subplot(subplot)
        subplot+=1
        plt.plot(x_range,result)
        samples = sample_gaussian_mixture([0.3, 0.5, 1], [0, -1, 1.5], [0.2, 0.3, 0.5], n_samples_tot[i])
        plt.hist(samples, 100, density=True)


    plt.savefig("./00_introduction/3_2_1.png")
    plt.show()




if __name__ == '__main__':

    print(normal(np.array([-1,0,1]), 1, 0))

    print(normal(0,1,0))
 

    plot_normal(0.5, 0, -2, 2)
    _plot_three_normals()
    normal_mixture(np.linspace(-5, 5, 5), [0.5, 0.25, 1], [0, 1, 1.5], [1/3, 1/3, 1/3])

    normal_mixture(np.linspace(-2, 2, 4), [0.5], [0], [1])

    _compare_components_and_mixture()

    print(sample_gaussian_mixture([0.1, 1], [-1, 1], [0.9, 0.1],4))
    print(sample_gaussian_mixture([0.1, 1, 1.5], [1, -1, 5], [0.1, 0.1, 0.8], 10))

    _plot_mixture_and_samples()
