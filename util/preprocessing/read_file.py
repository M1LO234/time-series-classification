import pandas as pd
import numpy as  np
from scipy.io import arff
from io import StringIO


def read_univariate(path: str, class_n: list, specific_files: list=None):
    data, _ = arff.loadarff(path)
    classes = get_classes(data)
    input_data = []
    if len(class_n) > 0:
        input_data = [np.array(list(f)[:-1], dtype=float) for f in data if f[-1].decode('utf-8') in [classes[c_id] for c_id in class_n]]
        labels = [f[-1].decode('utf-8') for f in data if f[-1].decode('utf-8') in [classes[c_id] for c_id in class_n]]
    else:
        input_data = [np.array(list(f)[:-1], dtype=float) for f in data]
        labels = [f[-1].decode('utf-8') for f in data]
    if specific_files:
        try:
            input_data = [input_data[i] for i in specific_files]
            labels = [labels[i] for i in specific_files]
        except IndexError:
            print(f'Files not found in the dataset: {specific_files}')
    
    return list(np.array(input_data).reshape((len(input_data), 1, -1))), labels

def read_multivariate(path: str, class_n: list, specific_files: list=None):
    data, _ = arff.loadarff(path)
    classes =get_classes(data)
    input_data, labels = [], []

    # list of dim(rows, variables) ndarray
    if len(class_n) > 0:
        sub_data = [f for f in data if f[-1].decode('utf-8') in [classes[c_id] for c_id in class_n]] #class types
    else:
        sub_data = [f for f in data]
    if specific_files:
        try:
            sub_data = [sub_data[i] for i in specific_files]
        except IndexError:
            print(f'Files not found in the dataset: {specific_files}')
    for sample in sub_data:
        s = np.transpose(np.array([np.array(list(dim)) for dim in sample[0]]))
        # s = [list(dim) for dim in sample[0]]
        input_data.append(s)
        labels.append(sample[-1].decode('utf-8'))

    classes_dict = {}
    class_val = 0
    for c_i in range(len(labels)):
        if labels[c_i] not in list(classes_dict.keys()):
            classes_dict[labels[c_i]] = class_val
            class_val += 1
    labels = [classes_dict[c] for c in labels]
    
    return input_data, labels

def get_classes(data):
    classes = list(set([cl[-1].decode('utf-8') for cl in data]))
    return sorted(classes)