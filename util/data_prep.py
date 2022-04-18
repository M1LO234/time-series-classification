import csv
import os
import random
import numpy as np
from .preprocessing.read_file import read_multivariate, read_univariate
from .preprocessing.get_multivariate import expand_series


def rescale(min, max, min_max_scale):
    if min_max_scale:
        return lambda x: min_max_scale[0] + np.subtract(x, min) * (min_max_scale[1]-min_max_scale[0]) / np.subtract(max, min)
    else:
        return lambda x: np.subtract(x, min) / np.subtract(max, min)

def import_and_transform(train_files, test_file, train_path, test_path, classif, classiftrain=None, sep=',', header=None, min_max_scale=None, rescLimits=None):
    model_input_train = []

    for t in train_files:
        curr_train_class = classif
        if classiftrain:
            curr_train_class = classiftrain
        with open(f'{train_path}/{curr_train_class}/{t}', newline='') as csv_file:
            model_input_train.append(np.array(list(csv.reader(csv_file))).astype(np.float))
    with open(f'{test_path}/{classif}/{test_file}', newline='') as csv_file:
        model_input_test = np.array(list(csv.reader(csv_file))).astype(np.float)

    model_input = np.concatenate((model_input_test, model_input_train[0]))
    for i, t in enumerate(model_input_train):
        if i == 0:
            continue
        model_input = np.concatenate((model_input, t))

    if rescLimits:
        max = np.array(rescLimits[1])
        min = np.array(rescLimits[0])
    else:
        max = model_input.max(0)
        min = model_input.min(0)

    for t in model_input_train:
        model_input_train = rescale(min, max, min_max_scale)(model_input_train)

    return model_input_train, rescale(min, max, min_max_scale)(model_input_test)


def import_from_dataset(amount=1, train_path='UWaveGestureLibrary/Train', test_path='UWaveGestureLibrary/Test',
                      classif=1, specificFiles=None, specTestFile=None, min_max_scale=None, rescLimits=None):
    
    if specificFiles != None:
        if isinstance(specificFiles, list):
            train_files = specificFiles
        else:
            train_files = [specificFiles]
    else:
        train_files = random.sample(os.listdir(f'{train_path}/{classif}'), amount)
    
    if specTestFile != None:
        test_files = [specTestFile+".csv"]
    else:
        test_files = random.sample(os.listdir(f'{test_path}/{classif}'), 1)

    train_series_set, test_series = import_and_transform(train_files, test_files[0], train_path, test_path, classif, min_max_scale=min_max_scale, rescLimits=rescLimits)
    return train_series_set, test_series, train_files, test_files

def import_from_arff(train_path, test_path, amount=1, classif=1, dims=1, specificFiles=None, specTestFile=None, min_max_scale=None, rescLimits=None):
    if dims > 1:
        test_files_amount = len(read_multivariate(test_path, [classif]))
    elif dims == 1:
        test_files_amount = len(read_univariate(test_path, [classif]))

    if specificFiles:
        if isinstance(specificFiles, list):
            spec_train_files = [int(sF) for sF in specificFiles]
        else:
            spec_train_files = [int(specificFiles)]
    else:
        spec_train_files = None
    if specTestFile:
        spec_test_file = [int(specTestFile)]
    else:
        spec_test_file = [random.randint(0, test_files_amount-1)]

    # read time series from file
    if dims > 1:
        train_series_set = read_multivariate(train_path, [classif], spec_train_files)
        test_series_set = read_multivariate(test_path, [classif], spec_test_file)
    elif dims == 1:
        train_series_set = expand_series(read_univariate(train_path, [classif], spec_train_files))
        test_series_set = expand_series(read_univariate(test_path, [classif], spec_test_file))

    # rescaling
    all_series = np.concatenate(([tr for tr in train_series_set], test_series_set))
    sc_min, sc_max = np.min(all_series), np.max(all_series)

    for tr_id, tr in enumerate(train_series_set):
        train_series_set[tr_id] = rescale(sc_min, sc_max, min_max_scale)(tr)

    return np.array(train_series_set), rescale(sc_min, sc_max, min_max_scale)(test_series_set[0]), spec_train_files, spec_test_file

    
