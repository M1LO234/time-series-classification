import pandas as pd
import numpy as  np
from scipy.io import arff
from io import StringIO

def read_univariate(path: str, class_n: list, specific_files: list=None):
    data, _ = arff.loadarff(path)
    classes = get_classes(data)
    input_data = []
    input_data = [np.array(list(f)[:-1], dtype=float) for f in data if f[-1].decode('utf-8') in [classes[c_id] for c_id in class_n]]
    if specific_files:
        try:
            input_data = [input_data[i] for i in specific_files]
        except IndexError:
            print(f'Files not found in the dataset: {specific_files}')
    return input_data

def read_multivariate(path: str, class_n: list, specific_files: list=None):
    data, _ = arff.loadarff(path)
    classes =get_classes(data)
    input_data = []

    # list of dim(rows, variables) ndarray
    sub_data = [f for f in data if f[-1].decode('utf-8') in [classes[c_id] for c_id in class_n]] #class types
    if specific_files:
        try:
            sub_data = [sub_data[i] for i in specific_files]
        except IndexError:
            print(f'Files not found in the dataset: {specific_files}')
    for sample in sub_data:
        s = np.transpose(np.array([np.array(list(dim)) for dim in sample[0]]))
        input_data.append(s)
    return input_data

def get_classes(data):
    classes = list(set([cl[-1].decode('utf-8') for cl in data]))
    return sorted(classes)