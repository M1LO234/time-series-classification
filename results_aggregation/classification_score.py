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

def perform_classification(path):
    files = [f for f in os.listdir(path) if '.csv' in f]
    dfs = dict()
    for f in files:
        curr_file = os.path.join(path, f)
        curr_df = pd.read_csv(curr_file, sep=';')
        num_cols = [c for c in curr_df.columns if 'min' in c or 'mean' in c]
        for col in num_cols:
            curr_df[col] = curr_df[col].str.replace(',', '.')
            curr_df[col] = curr_df[col].astype('float')
        train_class = curr_df['train class'][0]
        dfs[f'{train_class}'] = curr_df
    df_shape = curr_df.shape
    train_classes = list(dfs.keys())

    # classification
    unique_confs = np.unique(dfs[train_classes[0]]['conf'].values)
    all_preds, real_vals, iter, winning_confs = [], [], [], []
    test_classes = np.unique(dfs[train_classes[0]]['test class'].values)
    for uni in unique_confs:
       for cl in test_classes:
            iter.append((uni, cl))

    # for i, conf in enumerate(iter):
    for conf in range(df_shape[0]):
        rmse, mpe, max_pe = [], [], []
        for df_id, df_key in enumerate(train_classes):
            rmse.append(dfs[df_key].iloc[conf]['min rmse'])
            mpe.append(dfs[df_key].iloc[conf]['min mpe'])
            max_pe.append(dfs[df_key].iloc[conf]['min max_pe'])
            # rmse.append(dfs[df_key].loc[(dfs[df_key]['conf'] == conf[0]) & (dfs[df_key]['test class'] == int(conf[1]))]['min rmse'].mean())
            # mpe.append(dfs[df_key].loc[(dfs[df_key]['conf'] == conf[0]) & (dfs[df_key]['test class'] == int(conf[1]))]['min mpe'].mean())
            # max_pe.append(dfs[df_key].loc[(dfs[df_key]['conf'] == conf[0]) & (dfs[df_key]['test class'] == int(conf[1]))]['min max_pe'].mean())
       
        best_rmse, best_mpe, best_max_pe = np.argmin(rmse), np.argmin(mpe), np.argmin(max_pe)
        counts = np.bincount([best_rmse, best_mpe, best_max_pe])
        all_preds.append(int(train_classes[np.argmax(counts)]))
        # real_vals.append(dfs[train_classes[0]].loc[(dfs[df_key]['conf'] == conf[0]) & (dfs[df_key]['test class'] == int(conf[1]))]['test class'].values[0]) 
        real_vals.append(dfs[train_classes[0]].iloc[conf]['test class']) 
        if int(train_classes[np.argmax(counts)]) == dfs[train_classes[0]].iloc[conf]['test class']:
            winning_confs.append(dfs[train_classes[0]].iloc[conf]['conf'])

    print(Counter(winning_confs))
    print(accuracy(all_preds, real_vals))