import os
import pandas as pd
import numpy as np
from collections import Counter

def accuracy(y_test, pred):
    sum, lenght = 0, len(pred)
    for r in range(lenght):
        if pred[r] == y_test[r]:
            sum += 1
    return sum/lenght

def perform_classification(dfs):
    df_shape = dfs[list(dfs.keys())[0]].shape
    train_classes = list(dfs.keys())

    # classification
    unique_confs = np.unique(dfs[train_classes[0]]['conf'].values)
    all_preds, real_vals, iter, winning_confs = [], [], [], []
    test_classes = np.unique(dfs[train_classes[0]]['test class'].values)
    test_files = np.unique(dfs[train_classes[0]]['test file'].values)
    for cl_tr in list(dfs.keys()):
        for conf in unique_confs:
            iter.append([cl_tr, conf])
            
    results_based_on = ['min mpe', 'mean mpe', 'min rmse']
    for te_f in test_files:
        for cl in test_classes:
            class_results = []
            for c in enumerate(iter):
                curr_res_to_be_votes_on = []
                curr_df = dfs[f'{c[1][0]}'].loc[dfs[f'{c[1][0]}']['conf'] == c[1][1]]
                curr_errors = curr_df[results_based_on]\
                    .loc[(curr_df['test class'] == cl) & (curr_df['test file'] == te_f)]\
                        .values[0]
                curr_res_to_be_votes_on.append(curr_errors)
            print(curr_res_to_be_votes_on)
            return

    # for i, conf in enumerate(iter):
    # # for conf in range(df_shape[0]):
    #     rmse, mpe, max_pe = [], [], []
    #     for df_id, df_key in enumerate(train_classes):
    #         rmse.append(dfs[df_key].iloc[conf]['min rmse'])
    #         mpe.append(dfs[df_key].iloc[conf]['min mpe'])
    #         max_pe.append(dfs[df_key].iloc[conf]['min max_pe'])
    #         # rmse.append(dfs[df_key].loc[(dfs[df_key]['conf'] == conf[0]) & (dfs[df_key]['test class'] == int(conf[1]))]['min rmse'].mean())
    #         # mpe.append(dfs[df_key].loc[(dfs[df_key]['conf'] == conf[0]) & (dfs[df_key]['test class'] == int(conf[1]))]['min mpe'].mean())
    #         # max_pe.append(dfs[df_key].loc[(dfs[df_key]['conf'] == conf[0]) & (dfs[df_key]['test class'] == int(conf[1]))]['min max_pe'].mean())
       
    #     best_rmse, best_mpe, best_max_pe = np.argmin(rmse), np.argmin(mpe), np.argmin(max_pe)
    #     counts = np.bincount([best_rmse, best_mpe, best_max_pe])
    #     all_preds.append(int(train_classes[np.argmax(counts)]))
    #     # real_vals.append(dfs[train_classes[0]].loc[(dfs[df_key]['conf'] == conf[0]) & (dfs[df_key]['test class'] == int(conf[1]))]['test class'].values[0]) 
    #     real_vals.append(dfs[train_classes[0]].iloc[conf]['test class']) 
    #     if int(train_classes[np.argmax(counts)]) == dfs[train_classes[0]].iloc[conf]['test class']:
    #         winning_confs.append(dfs[train_classes[0]].iloc[conf]['conf'])

    print(Counter(winning_confs))
    print(accuracy(all_preds, real_vals))


agg_res_dir = '/Users/miloszwrzesien/Development/cognitiveMaps/new_fcm_module/fcm_results_agg'
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

dfs = get_results_to_compare(agg_res_dir)
perform_classification(dfs)