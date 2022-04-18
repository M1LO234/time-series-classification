import numpy as np
import scipy.optimize as optimize
from lmfit import Parameters, Minimizer


def opt_lmfit(fcm_weights, func, agg_weights=None):
    if type(agg_weights) == np.ndarray:
        flat_weights = np.concatenate((fcm_weights.flatten(), agg_weights.flatten()), axis=None)
    else:
        flat_weights = fcm_weights.flatten()

    params = Parameters()
    np.fromiter(map(lambda x: params.add(f'w{x[0]}', value=x[1], min=-1, max=1), enumerate(flat_weights)), dtype=float)

    fitter = Minimizer(func, params)
    result = fitter.minimize(method='nelder')
    n = fcm_weights.shape[0]

    err = func(result.params)
    fcm_weights = np.reshape(np.fromiter([result.params[f'w{i}'] for i in range(n * n)], dtype=float), (n, n))
    if type(agg_weights) == np.ndarray:
        agg_weights = np.reshape(np.fromiter([result.params[f'w{i}'] for i in range(n * n, len(flat_weights))], dtype=float), (agg_weights.shape[0], n))

    return fcm_weights, agg_weights, err


def lmfit_inner(transformation, win_trans, fcm_weights, x, y_test, error, agg_weights=None):
    n = fcm_weights.shape[0]

    def func(w):
        fw = np.reshape(np.fromiter([w[f'w{i}'] for i in range(n * n)], dtype=float), (n, n))
        aw = None
        if type(agg_weights) == np.ndarray:
            aw = np.reshape(np.fromiter([w[f'w{i}'] for i in range(n * n, len(w))], dtype=float), (agg_weights.shape[0], n))
        pred = calc(transformation, win_trans, fw, x, aw)
        return error(y_test, pred)

    fcm_weights, agg_weights, err = opt_lmfit(fcm_weights, func, agg_weights)
    return fcm_weights, agg_weights, err


def lmfit_outer(
        transformation, win_trans,
        fcm_weights,
        time_series, step, window,
        error, agg_weights=None
):
    n = fcm_weights.shape[0]

    def func(w):
        fw = np.reshape(np.fromiter([w[f'w{i}'] for i in range(n * n)], dtype=float), (n, n))
        aw = None
        if type(agg_weights) == np.ndarray:
            aw = np.reshape(np.fromiter([w[f'w{i}'] for i in range(n * n, len(w))], dtype=float), (agg_weights.shape[0], n))

        yts, ys = calc_all(time_series, step, window, transformation, win_trans, fw, aw)
        return error(ys, yts)


    fcm_weights, agg_weights, e = opt_lmfit(fcm_weights, func, agg_weights)
    return fcm_weights, agg_weights, e

def calc(transformation, win_trans, weights, x, agg_weights=None):
    if type(agg_weights) == np.ndarray:
        return transformation(np.matmul(weights,np.einsum("ij,ij->j", agg_weights, x)))
    else:
        return transformation(np.matmul(weights,win_trans(x)))


def calc_all(time_series, step, window, transformation, win_trans, weights, agg_weights=None):
    pred = np.array([])
    y_test = np.array([])

    for step in step(time_series, window):
        curr_pred = [calc(transformation, win_trans, weights, step['x'], agg_weights)]
        pred = np.concatenate((pred, curr_pred), axis=None)
        y_test = np.concatenate((y_test, step['y']), axis=None)
    return pred, y_test


def inner(time_series, window, step, transformation, win_trans, weights, error, agg_weights=None):
    error_max = -1
    for step in step(time_series, window):
        weights, agg_weights, e = lmfit_inner(
            transformation, win_trans,
            weights,
            step['x'], step['y'],
            error, agg_weights
        )

        if error_max < e:
            error_max = e

    return weights, agg_weights, error_max


def outer(time_series, window, step, transformation, win_trans, weights, error, agg_weights=None):
    weights, agg_weights, e = lmfit_outer(
        transformation, win_trans,
        weights,
        time_series, step, window,
        error, agg_weights
    )

    return weights, agg_weights, e
