from genericpath import exists
import os
import json
import pandas as pd
import numpy as np

def repl_commas(val):
    return str(val).replace('.', ',')

def voted_results(arr): #
    indexes = np.argmin(arr, axis=0)
    counts = np.bincount(indexes)
    return np.argmax(counts)

def iterate_over_dirs(main_dir, out_dir, range_dirs: list=None):
    list_dirs = [cd for cd in os.listdir(main_dir) if '.' not in cd]
    if range_dirs:
        list_dirs = list_dirs[range_dirs[0]:range_dirs[1]]
    dirs = [os.path.join(main_dir, cd) for cd in list_dirs if '.' not in cd]
    
    for dir_id, dir in enumerate(dirs):
        conf_names = [cd for cd in os.listdir(dir) if '.' not in cd]
        conf_names = sorted(conf_names)
        for conf_name in conf_names:
            conf_dir = os.path.join(dir, conf_name)
            json_name = [f_name for f_name in os.listdir(conf_dir) if '.json' in f_name][0]
            json_path = os.path.join(conf_dir, json_name)
            curr_out_dir = os.path.join(out_dir, list_dirs[dir_id])
            aggregate_from_json(json_path, curr_out_dir, conf_name)

# todo: IN - json path, out_path, OUT - csv with all results, txt describing configurations
def aggregate_from_json(json_path, output_path, configuration_name):
    with open(json_path, 'r') as f:
        con = json.load(f)

    curr_dir = output_path+'/'
    if not os.path.exists(curr_dir):
        os.mkdir(curr_dir)

    out_csv_path = output_path+'/'+output_path.split('/')[-1]+'.csv'
    out_txt_path = output_path+'/'+output_path.split('/')[-1]+'_configs.csv'
    out_summary_path = output_path+'/'+output_path.split('/')[-1]+'_summary.csv'
    no_files = len(con['files']['training'])

    # save configuration description
    conf_values = list(con['config'].keys())
    cols_conf = ['conf', 'no. train files']+conf_values
    curr_conf_values = [configuration_name, no_files] + [str(con['config'][k]) for k in conf_values]
    df = pd.DataFrame([curr_conf_values], columns=cols_conf)
    df.to_csv(out_txt_path, mode='a', header=not os.path.exists(out_txt_path) , sep=';', index=False)

    cols = ['conf', 'no. train files', 'train file(s)', 'train class', 'test file', 'test class', 'min rmse', 'mean rmse', 'min mpe', 'mean mpe', 'min max_pe', 'mean max_pe']
    classes_tested = list(con['test results'].keys())
    conf_results, calc_score = [], []
    class_of_train = con['files']['class']
    train_files = ','.join(['_'+str(tr) for tr in con['files']['training']])
    for class_id, class_n in enumerate(classes_tested):
        test_files_for_curr_class = list(con['test results'][class_n].keys())
        calc_score.append([])
        for test_file in test_files_for_curr_class:
            rmse = con['test results'][class_n][test_file]['rmse'] #todo: reading multiple values (min, mean, max)
            mpe = con['test results'][class_n][test_file]['mpe']
            max_pe = con['test results'][class_n][test_file]['max_pe']

            # classifier scores
            # calc_score[class_id].append([np.array(rmse).mean(), np.array(mpe).mean(), np.array(max_pe).mean()])
            calc_score[class_id].append([np.array(rmse).min(), np.array(mpe).min(), np.array(max_pe).min()])

            curr_line = [configuration_name, no_files, train_files, class_of_train, test_file, \
                class_n, repl_commas(np.array(rmse).min()), repl_commas(np.array(rmse).mean()), \
                    repl_commas(np.array(mpe).min()), repl_commas(np.array(mpe).mean()), \
                    repl_commas(np.array(max_pe).min()), repl_commas(np.array(max_pe).mean())]
            conf_results.append(curr_line)
        calc_score[class_id] = np.mean(calc_score[class_id], axis=0)

    conf_df = pd.DataFrame(conf_results, columns=cols)
    conf_df.to_csv(out_csv_path, mode='a', header=not os.path.exists(out_csv_path), sep=';', index=False)

    # prediction based on voting
    pred_class = classes_tested[voted_results(calc_score)]
    pred_df = pd.DataFrame([[configuration_name, class_of_train, pred_class]], columns=['configuration', 'real class', 'prediction'])
    pred_df.to_csv(out_summary_path, mode='a', header=not os.path.exists(out_summary_path), sep=';', index=False)

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
    all_preds, real_vals, iter = [], [], []
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
    print(accuracy(all_preds, real_vals))




out_dir = '/Users/miloszwrzesien/Development/cognitiveMaps/new_fcm_module/fcm_results_agg'
main_dir = '/Users/miloszwrzesien/Development/cognitiveMaps/new_fcm_module/fcm_results'
class_path = '/Users/miloszwrzesien/Development/cognitiveMaps/new_fcm_module/fcm_results_agg/cls'
# iterate_over_dirs(main_dir, out_dir)
perform_classification(class_path)