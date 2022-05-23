import numpy as np
# input: univariate time series
# returns multivariate time series dim(N, d), where d<1

def expand_series(data):
    pass

def get_lag(sample, order):
    vals = []
    if order == 1:
        for i in range(1, sample.shape[0]):
            vals.append(sample[i]-sample[i-1])
    elif order == 2:
        for i in range(2, sample.shape[0]):
            vals.append(sample[i]-2*sample[i-1]+sample[i-2])
    elif order == 3:
        for i in range(3, sample.shape[0]):
            vals.append(sample[i]-3*sample[i-2]+3*sample[i-1]-sample[i-2])
    return np.array(vals)