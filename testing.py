import json
import argparse
from locale import currency
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from dtw import *

from util.fcm_util import data_prep, metrics, squashing_functions, weight_optimization, window_steps, window_transforms

# tmp
from util.preprocessing.read_file import read_multivariate

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

def test_fcm(ts, test_file=None, class_test=None, arff=True, use_dtw=False):
    if not os.path.exists(ts):
        print("Output folder not found or no test id given")
        return

    with open(ts, "r") as f:
        train_summary = json.load(f)

    class_train = train_summary['files']['class']
    if class_test != None:
        class_test = int(class_test)

    if arff:
        train_series_set, test_series, tr, te = data_prep.import_from_arff(train_path=train_summary['files']['train path'],
                                                                test_path=train_summary['files']['test path'],
                                                                class_train=class_train, 
                                                                class_test=class_test,
                                                                dims=train_summary['files']['dimensions'],
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

    # tmp
    tmp_saved_preds, tmp_real_vals = [], []

    overall_results = dict()
    for i, series in enumerate(series):
        test_errors = {'rmse': [], 'mpe': [], 'max_pe': []}

        # tmp
        saved_preds, real_vals = [], []

        for step_i in getattr(window_steps, train_summary['config']['step'])(series, train_summary['config']['window size']):
            yt = weight_optimization.calc(getattr(squashing_functions, train_summary['config']['transformation function'])(),
                            window_transforms.mean_transform,
                            np.array(train_summary['weights']['fcm']),
                            step_i['x'],
                            agg_weights)

            # tmp
            if use_dtw:
                saved_preds.append(yt)
                real_vals.append(step_i['y'])

            test_errors['rmse'].append(metrics.rmse(step_i['y'], yt))
            test_errors['mpe'].append(metrics.mpe(step_i['y'], yt))
            test_errors['max_pe'].append(metrics.max_pe(step_i['y'], yt))

        if use_dtw:
            dtw_dist = dtw(saved_preds, real_vals)
            test_errors['dtw'] = dtw_dist.distance

        # return
        # tmp: saving preds
        # tmp_saved_preds.append(saved_preds)
        # tmp_real_vals.append(real_vals)

        overall_results[i] = test_errors
    # pred = np.array(tmp_saved_preds[0])
    # real = np.array(tmp_real_vals[0])
    # df = np.concatenate((pred, real), axis=1)
    # df_res = pd.DataFrame(df, columns=['pred_x', 'pred_y', 'pred_z', 'real_x', 'real_y', 'real_z'])
    # csv_path = ts.split('.')[:1][0]+f'_ct_{class_test}_tf{te[0]}.csv'
    # df_res.to_csv(csv_path)

    curr_test_class_key = f'{class_test if class_test != None else class_train}'
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

        if use_dtw:
            train_summary[res_key][curr_test_class_key][file_key] = {
                'rmse': overall_results[i]["rmse"],
                'mpe': overall_results[i]["mpe"],
                'max_pe': overall_results[i]["max_pe"],
                'dtw': overall_results[i]["dtw"]
            }
        else:
            train_summary[res_key][curr_test_class_key][file_key] = {
                'rmse': overall_results[i]["rmse"],
                'mpe': overall_results[i]["mpe"],
                'max_pe': overall_results[i]["max_pe"]
            }

    with open(ts, 'w') as f:
        json.dump(train_summary, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fuzzy Cognitive Mapping testing')
    parser.add_argument('-ts', dest='ts', default='1607964137', type=str, help='Path to the .json file with weights')
    parser.add_argument('-tf', dest='tf', default=None, type=str, help='test file name (eg. "356.csv"')

    args = parser.parse_args()

    test_fcm(args.ts, args.tf)
