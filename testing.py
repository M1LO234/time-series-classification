import json
import argparse
import os
import numpy as np
from matplotlib import pyplot as plt

from util import data_prep, metrics, squashing_functions, weight_optimization, window_steps, window_transforms


def test_fcm(ts, test_file=None, arff=True,):
    if not os.path.exists(ts):
        print("Output folder not found or no test id given")
        return

    with open(ts, "r") as f:
        train_summary = json.load(f)

    if arff:
        train_series_set, test_series, tr, te = data_prep.import_from_arff(train_path=train_summary['files']['train path'],
                                                                test_path=train_summary['files']['test path'],
                                                                classif=train_summary['files']['class'], 
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
            train_summary["files"]["class"]
        )

    if train_summary['weights']['aggregation']:
        agg_weights = np.array(train_summary['weights']['aggregation'])
    else:
        agg_weights = None

    series = [test_series]
    series.extend(train_series_set)

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

    train_summary['test results'][f'{te[0]}'] = {
        'rmse': np.array(test_errors["rmse"]).mean(),
        'mpe': np.array(test_errors["mpe"]).mean(),
        'max_pe': np.array(test_errors["max_pe"]).mean()
    }

    with open(ts, 'w') as f:
        json.dump(train_summary, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fuzzy Cognitive Mapping testing')
    parser.add_argument('-ts', dest='ts', default='1607964137', type=str, help='Path to the .json file with weights')
    parser.add_argument('-tf', dest='tf', default=None, type=str, help='test file name (eg. "356.csv"')

    args = parser.parse_args()

    test_fcm(args.ts, args.tf)
