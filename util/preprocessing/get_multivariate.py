import numpy as np
# input: univariate time series
# returns multivariate time series dim(N, d), where d<1


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


def expand_series(data):
    out = []
    for sample in data:
        basic_dim = sample[0]
        first_lag = get_lag(basic_dim, 1)[1:]
        second_lag = get_lag(basic_dim, 2)
        original = basic_dim[2:]
        res = np.transpose(np.stack((original, first_lag, second_lag)))
        out.append(res)
    return out

def expand_dim_with_lags(series, lags_list):
    out, new_dims = [], []
    max_len = series[0].shape[0] - np.max(lags_list)
    for sample in series:
        for dim in range(sample.shape[1]):
            for lag_degree in lags_list:
                    new_dims.append(get_lag(sample[:,dim], lag_degree)[:max_len])
        new_dims = np.array(new_dims).T
        out.append(np.append(sample[:max_len,:], new_dims, axis=1))
    return out
        

    