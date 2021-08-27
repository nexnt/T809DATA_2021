import numpy as np

def prior(targets: np.ndarray, classes:list) -> np.ndarray:
    no = len(targets)
    result = []
    for i in range(len(classes)):
        print(targets.count(classes[i]))
        result.append((targets.count(classes[i]))/no)
    return np.array(result)

print(prior([2, 1,1, 1,0], [0, 1]))

