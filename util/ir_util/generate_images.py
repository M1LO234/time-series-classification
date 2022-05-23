import numpy as np
import pandas as pd
from ..preprocessing.get_multivariate import get_lag

def standarize(sample):
    return (sample - np.mean(sample))/np.std(sample)

def get_rg(time_series, lag_order, brigthness_lvl):
    lags, lags_st = [], []
    raw_data_norm, features = [], []
    for i, sample in enumerate(time_series):
        first_lag = get_lag(sample, lag_order)

        # normalization of image
        if np.min(sample) < 0:
            sample = sample+abs(np.min(sample))
        sample = sample / np.max(sample)
        
        first_lag_st = standarize(first_lag)
        lags.append(first_lag)
        lags_st.append(first_lag_st)
        raw_data_norm.append(sample[lag_order+1:])

        lags_dicr = pd.cut(lags[i], brigthness_lvl, labels=False) - brigthness_lvl/2
        lags_dicr_rgb = []
        lvl = 255/brigthness_lvl
        for elem in lags_dicr:
            if elem > 0:
                lags_dicr_rgb.append([0, lvl*elem, 0])
            elif elem < 0:
                lags_dicr_rgb.append([-elem*lvl, 0, 0])
            else:
                lags_dicr_rgb.append([0, 0, 0])
        lags_dicr_rgb = np.array(lags_dicr_rgb, dtype=int)
        final_img = np.array([lags_dicr_rgb for v in range(len(lags_dicr_rgb))])
        features.append(final_img)
    return features

def get_train_test(method):
    pass