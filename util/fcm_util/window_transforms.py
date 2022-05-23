import numpy as np

 #TODO: transformation module; out-> (N,); N - equal to number of nodes

def mean_transform(x):
    return np.mean(x, axis=0)