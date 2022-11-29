import statistics
import os
import pandas as pd
import copy
import numpy as np
from collections import Counter
from aggregate_classification import voted_results

def accuracy(y_test, pred):
    sum, lenght = 0, len(pred)
    for r in range(lenght):
        if pred[r] == y_test[r]:
            sum += 1
    return sum/lenght

def perform_classification(dfs, col_substring, main_path, line_by_line=True, individual=False, ovo=False, pred_from_ovo=False):
    train_classes = sorted(list(dfs.keys()))
    if ovo:
        results_based_on = ['min rmse', 'mean rmse', 'min mpe', 'mean mpe', 'min max_pe', 'mean max_pe', 'dtw']
    results_based_on = ['min rmse', 'mean rmse', 'min mpe', 'mean mpe', 'min max_pe', 'mean max_pe']
    results_based_on_metric = [c for c in results_based_on if col_substring in c]
    req_columns = ['conf', 'no. train files', 'train file(s)', 'test file', 'test class']
    df_results = pd.DataFrame([], columns=req_columns+['prediction'])

    if line_by_line:
        for line in range(dfs[list(dfs.keys())[0]].shape[0]):
            curr_line_results = []
            for tr_class in train_classes:
                curr_line = dfs[tr_class][results_based_on_metric].iloc[line].values
                curr_line_results.append(curr_line)
            curr_res_line_to_df = dict(dfs[tr_class][req_columns].iloc[line])

            if ovo:
                curr_res_line_to_df['ovo_pred'] = int(ovo_binary_classification(train_classes, values=curr_line_results))

            if individual:
                for metric_id in range(curr_line_results[0].shape[0]):
                    curr_res_line_to_df[f'pred_{results_based_on_metric[metric_id]}'] = int(np.argmin(np.array(curr_line_results)[:, metric_id]))

            curr_res_line_to_df['prediction'] = voted_results(curr_line_results)
            df_results = df_results.append(curr_res_line_to_df, ignore_index=True)
    # else:
    #     for 

    pred_column = 'prediction' if not pred_from_ovo else 'ovo_pred'

    df_conf_results = pd.DataFrame([], columns=['conf', 'accuracy'])
    for conf in np.unique(df_results['conf'].values):
        curr_conf_test_set = df_results['test class'].loc[df_results['conf'] == conf].values
        curr_conf_preds = df_results[pred_column].loc[df_results['conf'] == conf].values
        curr_conf_acc = accuracy(curr_conf_test_set, curr_conf_preds)
        df_conf_results = df_conf_results.append({'conf': conf, 'accuracy': curr_conf_acc}, ignore_index=True)

    res_accuracy = int(accuracy(df_results['test class'].values, df_results[pred_column].values) * 100)
    res_file_name_base = res_file_name = dfs[tr_class]['conf'].values[0].split('class')[0]
    res_file_name = res_file_name_base + f'acc_{res_accuracy}perc.csv'
    out_path = os.path.join(main_path, res_file_name)
    out_path_confs = os.path.join(main_path, res_file_name_base[:-1]+'.csv')
    df_conf_results.to_csv(out_path_confs, mode='w', sep=';', index=False)
    # print(df_results.head())
    # df_results[df_results.columns[-(len(results_based_on_metric)+1):]] = df_results[df_results.columns[-(len(results_based_on_metric)+1):]].astype('int')
    df_results.to_csv(out_path, mode='w', sep=';', index=False)
    return df_results

def get_results_to_compare(main_dir_path):
    dfs = dict()
    dirs = [f for f in os.listdir(main_dir_path) if '.' not in f]
    excluded_file_postfixes = ['_configs', '_summary']
    for res_dir in dirs:
        curr_res_dir = os.path.join(main_dir_path, res_dir)
        curr_res_file = [f for f in os.listdir(curr_res_dir) if all(x not in f for x in excluded_file_postfixes) and '.csv' in f][0]
        curr_res_file_path = os.path.join(main_dir_path, curr_res_dir, curr_res_file)
        curr_df = pd.read_csv(curr_res_file_path, sep=';')
        
        num_cols = [c for c in curr_df.columns if any(x in c for x in ['min', 'mean'])]
        for col in num_cols:
            curr_df[col] = curr_df[col].str.replace(',', '.')
            curr_df[col] = curr_df[col].astype('float')
        train_class = curr_df['train class'][0]
        dfs[f'{train_class}'] = curr_df
    return dfs

def ovo_binary_classification(train_classes, values):
    predictions = []
    curr_classes = copy.deepcopy(train_classes)
    for _, class_i in enumerate(train_classes):
        curr_classes.remove(class_i)
        for _, class_j in enumerate(curr_classes):
            win_id = voted_results(np.array([values[int(class_i)], values[int(class_j)]]))
            predictions.append(int([class_i, class_j][win_id]))
    return statistics.mode(predictions)



agg_res_dir = '/Users/miloszwrzesien/Development/cognitiveMaps/new_fcm_module/fcm_results_agg'
dfs = get_results_to_compare(agg_res_dir)
# df_results = perform_classification(dfs, 'dtw', agg_res_dir, True, True, True, True)
df_results = perform_classification(dfs, '', agg_res_dir, True)

# path_to_df_res = '/Users/miloszwrzesien/Development/cognitiveMaps/new_fcm_module/fcm_results_agg/25_10_19_29_ERing_acc_18perc.csv'
# df_results = pd.read_csv(path_to_df_res, sep=';')
# ovo_binary_classification(dfs, df_results, '', agg_res_dir)