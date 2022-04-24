import json
import argparse
from locale import currency
import os
import numpy as np
from matplotlib import pyplot as plt

from util import data_prep, metrics, squashing_functions, weight_optimization, window_steps, window_transforms

# helper
def check_dict_for_key(key, dictionary, remove_existing_val=False):
    if key not in dictionary:
        dictionary[key] = {}
        return dictionary
    elif remove_existing_val:
        dictionary[key] = {}
        return dictionary
    else:
        return dictionary

def test_fcm(ts, test_file=None, class_test=None, arff=True):
    if not os.path.exists(ts):
        print("Output folder not found or no test id given")
        return

    with open(ts, "r") as f:
        train_summary = json.load(f)

    class_train = train_summary['files']['class']
    if class_test:
        class_test = int(class_test)

    if arff:
        train_series_set, test_series, tr, te = data_prep.import_from_arff(train_path=train_summary['files']['train path'],
                                                                test_path=train_summary['files']['test path'],
                                                                class_train=class_train, 
                                                                class_test=class_test,
                                                                dims=3, 
                                                                specificFiles=train_summary['files']['training'],
                                                                specTestFile=train_summary['files']['testing'] if test_file is None else test_file, 
                                                                min_max_scale=train_summary['config']['data normalization ranges'],
                                                                rescLimits=None)
    else:
        train_series_set, test_series = data_prep.import_and_transform(
            train_summary['files']['training'],
            train_summary['files']['testing'] if test_file is None else test_file,
            train_summary["files"]["train path"],
            train_summary['files']['test path'],
            class_train
        )

    if train_summary['weights']['aggregation']:
        agg_weights = np.array(train_summary['weights']['aggregation'])
    else:
        agg_weights = None

    series = [test_series]
    series.extend(train_series_set)

    overall_results = dict()
    for i, series in enumerate(series):
        test_errors = {'rmse': [], 'mpe': [], 'max_pe': []}
        for step_i in getattr(window_steps, train_summary['config']['step'])(series, train_summary['config']['window size']):
            yt = weight_optimization.calc(getattr(squashing_functions, train_summary['config']['transformation function'])(),
                            window_transforms.mean_transform,
                            np.array(train_summary['weights']['fcm']),
                            step_i['x'],
                            agg_weights)

            test_errors['rmse'].append(metrics.rmse(step_i['y'], yt))
            test_errors['mpe'].append(metrics.mpe(step_i['y'], yt))
            test_errors['max_pe'].append(metrics.max_pe(step_i['y'], yt))
        overall_results[i] = test_errors

    curr_test_class_key = f'{class_test if class_test else class_train}'
    if str(class_train) in list(train_summary['train results'].keys()):
        overall_results = [overall_results[0]]
    for i in range(len(overall_results)):
        if i == 0:
            res_key = 'test results'
            file_key = f'{te[0]}'
        else:
            res_key = 'train results'
            file_key = f'{tr[i-1]}'
            curr_test_class_key = class_train

        train_summary[res_key] = check_dict_for_key(curr_test_class_key, train_summary[res_key])
        train_summary[res_key][curr_test_class_key][file_key] = {
            'rmse': np.array(overall_results[i]["rmse"]).mean(),
            'mpe': np.array(overall_results[i]["mpe"]).mean(),
            'max_pe': np.array(overall_results[i]["max_pe"]).mean()
        }

    with open(ts, 'w') as f:
        json.dump(train_summary, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fuzzy Cognitive Mapping testing')
    parser.add_argument('-ts', dest='ts', default='1607964137', type=str, help='Path to the .json file with weights')
    parser.add_argument('-tf', dest='tf', default=None, type=str, help='test file name (eg. "356.csv"')

    args = parser.parse_args()

    test_fcm(args.ts, args.tf)
