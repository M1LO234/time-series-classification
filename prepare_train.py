import random
import numpy as np
from tqdm import trange

from util import window_transforms

def fcm_train(train_series_set, step, transition_func, error, mode, max_iter, performance_index,
            window, use_aggregation=False, get_best_weights=False, passing_files_method='random'):
    '''
        train_series_set: numpy array, dims(n_o_files, time_stamps, n_o_vars_in_series)
    '''

    errors = []
    fuzzy_nodes = train_series_set[0].shape[1]
    weights = np.random.rand(fuzzy_nodes, fuzzy_nodes)
    if use_aggregation:
        agg_weights = np.random.rand(window, fuzzy_nodes)
    else:
        agg_weights = None
    all_weights = []

    if passing_files_method == 'sequential':
        train_batch = train_series_set.reshape(1,-1,train_series_set.shape[2])[0]

    for _ in trange(max_iter, desc='model iterations', leave=True):
        if passing_files_method == 'random':
            train_batch = random.choice(train_series_set)

        weights, agg_weights, loop_error = mode(
            train_batch, window,
            step, transition_func(), window_transforms.mean_transform,
            weights, error, agg_weights
        )
        print("loop_error: ", loop_error)

        all_weights.append(weights)
        errors.append(loop_error)

        if loop_error <= performance_index:
            break
    
    if get_best_weights:
        weights = all_weights[errors.index(np.min(errors))]

    return weights, errors, loop_error
    
    