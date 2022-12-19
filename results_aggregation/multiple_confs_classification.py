import os
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm
sys.path.append("..")
from aggregate_classification import *
from sklearn.metrics import accuracy_score

configs_path = '/Users/miloszwrzesien/Development/cognitiveMaps/new_fcm_module/multiple_configs_results'
multiple_confs_output_path = '/Users/miloszwrzesien/Development/cognitiveMaps/new_fcm_module/multiple_configs_results/tests'

if __name__ == "__main__":
    sufix = str(sys.argv[1])
    res_type = int(sys.argv[2])

jsons_by_class = [[],[]]
for conf_main_dir in [c for c in os.listdir(configs_path) if 'tests' not in c]:
    curr_main = os.path.join(configs_path, conf_main_dir, 'fcm_results')
    next_listed = [c for c in os.listdir(curr_main) if os.path.isdir(os.path.join(curr_main, c))]
    next_listed.sort()
    for c_id, classes_dir in enumerate(next_listed):
        curr_class_dir = os.path.join(curr_main, classes_dir)
        curr_class_jsons = []
        for conf_dir in [c for c in os.listdir(curr_class_dir) if os.path.isdir(os.path.join(curr_class_dir, c))]:
            curr_class_conf_dir = os.path.join(curr_class_dir, conf_dir)
            curr_conf_json = [f for f in os.listdir(curr_class_conf_dir) if '.json' in f][0]
            curr_class_jsons.append(os.path.join(curr_class_conf_dir, curr_conf_json))
        jsons_by_class[c_id] += curr_class_jsons

dict_conf_names = {}

class0 = jsons_by_class[0]
class0.sort()
for conf_id, conf_file in enumerate(class0):
    dict_conf_names[f'c0-conf{conf_id}'] = conf_file.replace(configs_path,'')[1:]
    aggregate_from_json(conf_file, multiple_confs_output_path, f'c0-conf{conf_id}', multi_comparison=True)

class1 = jsons_by_class[1]
class1.sort()
for conf_id, conf_file in enumerate(class1):
    dict_conf_names[f'c1-conf{conf_id}'] = conf_file.replace(configs_path,'')[1:]
    aggregate_from_json(conf_file, multiple_confs_output_path, f'c1-conf{conf_id}', multi_comparison=True)

df0 = pd.read_csv(multiple_confs_output_path+'/c0.csv', sep=';', header=0)
df1 = pd.read_csv(multiple_confs_output_path+'/c1.csv', sep=';', header=0)

confs = np.unique(df0['conf'].values)
opp_confs = np.unique(df1['conf'].values)
results_based_on = ['min rmse', 'mean rmse', 'min mpe', 'mean mpe', 'min max_pe', 'mean max_pe']
relevant_cols_list = [[results_based_on, 'all_cols'], [[results_based_on[1]], 'rmse'], [[results_based_on[3]], 'mpe'], [[results_based_on[5]], 'max_pe'] ]

for r_id, relevant_cols in enumerate([relevant_cols_list[res_type]]):
    out_values = []
    for conf in tqdm(confs, desc='Aggregating...'): #class 0 confs
        test_files = np.unique(df0.loc[df0['conf'] == conf]['test file'].values)
        curr_df0 = df0.loc[df0['conf'] == conf]
        for opp_conf in opp_confs:
            curr_df1 = df1.loc[df1['conf'] == opp_conf]
            opp_test_files = np.unique(curr_df1['test file'].values)
            test_files_intersect = np.array(list(set(test_files).intersection(set(opp_test_files))))
            pred_test_values = []
            for test_file_name in test_files_intersect:
                for n_class in [0,1]:
                    curr_df0_line = curr_df0[relevant_cols[0]].loc[(curr_df0['test file'] == test_file_name) & (curr_df0['test class'] == n_class)].values
                    curr_df1_line = curr_df1[relevant_cols[0]].loc[(curr_df1['test file'] == test_file_name) & (curr_df1['test class'] == n_class)].values

                    curr_line_results = []
                    curr_line_results.append(curr_df0_line[0])
                    curr_line_results.append(curr_df1_line[0])
                    pred_test_values.append([n_class, voted_results(curr_line_results)])
            out_values.append([conf, opp_conf, str(accuracy_score(np.array(pred_test_values)[:,0], np.array(pred_test_values)[:,1])).replace('.',','),\
                dict_conf_names[conf], dict_conf_names[opp_conf]])
    out_df = pd.DataFrame(out_values, columns=['conf_c0', 'conf_c1', 'accuracy', 'conf_c0_path', 'conf_c1_path'])
    out_df.to_csv(multiple_confs_output_path+f'/{sufix}_{relevant_cols[1]}.csv')
