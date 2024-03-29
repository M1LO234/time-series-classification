import os
import json
import pandas as pd
import numpy as np

def repl_commas(val):
    return str(val).replace('.', ',')

def voted_results(arr):
    indexes = np.argmin(arr, axis=0)
    counts = np.bincount(indexes)
    return np.argmax(counts)

def iterate_over_dirs(main_dir, out_dir, range_dirs: list=None, use_dtw=False):
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
            aggregate_from_json(json_path, curr_out_dir, conf_name, use_dtw)

def aggregate_from_json(json_path, output_path, configuration_name, use_dtw=False, multi_comparison=False):
    with open(json_path, 'r') as f:
        con = json.load(f)

    curr_dir = output_path+'/'
    if not os.path.exists(curr_dir):
        os.mkdir(curr_dir)
    
    out_txt_path = output_path+'/'+output_path.split('/')[-1]+'_configs.csv'
    out_summary_path = output_path+'/'+output_path.split('/')[-1]+'_summary.csv'
    no_files = len(con['files']['training'])

    # save configuration description
    conf_values = list(con['config'].keys())
    cols_conf = ['conf', 'no. train files']+conf_values
    train_files = ','.join(['_'+str(tr) for tr in con['files']['training']])

    curr_conf_values = [configuration_name, train_files] + [str(con['config'][k]) for k in conf_values]
    df = pd.DataFrame([curr_conf_values], columns=cols_conf)

    if not multi_comparison:
        out_csv_path = output_path+'/'+output_path.split('/')[-1]+'.csv'
        df.to_csv(out_txt_path, mode='a', header=not os.path.exists(out_txt_path) , sep=';', index=False)
    else:
        out_csv_path = output_path+'/'+f'c{configuration_name[1]}.csv'

    if use_dtw:
        cols = ['conf', 'no. train files', 'train file(s)', 'train class', 'test file', 'test class', 'min rmse', 'mean rmse', 'min mpe', 'mean mpe', 'min max_pe', 'mean max_pe', 'dtw']
    else:
        cols = ['conf', 'no. train files', 'train file(s)', 'train class', 'test file', 'test class', 'min rmse', 'mean rmse', 'min mpe', 'mean mpe', 'min max_pe', 'mean max_pe']

    classes_tested = list(con['test results'].keys())
    conf_results, calc_score = [], []
    class_of_train = con['files']['class']
    for class_id, class_n in enumerate(classes_tested):
        test_files_for_curr_class = list(con['test results'][class_n].keys())
        calc_score.append([])
        for test_file in test_files_for_curr_class:
            if test_file in ['0', '99'] :
                continue

            rmse = con['test results'][class_n][test_file]['rmse']
            mpe = con['test results'][class_n][test_file]['mpe']
            max_pe = con['test results'][class_n][test_file]['max_pe']

            if use_dtw:
                dtw = con['test results'][class_n][test_file]['dtw']
                calc_score[class_id].append([np.array(rmse).mean(), np.array(mpe).mean(), np.array(max_pe).mean(), dtw])
                curr_line = [configuration_name, no_files, train_files, class_of_train, test_file, \
                class_n, repl_commas(np.array(rmse).min()), repl_commas(np.array(rmse).mean()), \
                    repl_commas(np.array(mpe).min()), repl_commas(np.array(mpe).mean()), \
                    repl_commas(np.array(max_pe).min()), repl_commas(np.array(max_pe).mean()), repl_commas(np.array(dtw).mean())]
            else:
            # classifier scores
                calc_score[class_id].append([np.array(rmse).mean(), np.array(mpe).mean(), np.array(max_pe).mean()])
            # calc_score[class_id].append([np.array(rmse).min(), np.array(mpe).min(), np.array(max_pe).min()])

                curr_line = [configuration_name, no_files, train_files, class_of_train, test_file, \
                    class_n, repl_commas(np.array(rmse).min()), repl_commas(np.array(rmse).mean()), \
                        repl_commas(np.array(mpe).min()), repl_commas(np.array(mpe).mean()), \
                        repl_commas(np.array(max_pe).min()), repl_commas(np.array(max_pe).mean())]
            conf_results.append(curr_line)
        calc_score[class_id] = np.mean(calc_score[class_id], axis=0)

    conf_df = pd.DataFrame(conf_results, columns=cols)
    conf_df.to_csv(out_csv_path, mode='a', header=not os.path.exists(out_csv_path), sep=';', index=False)

    # prediction based on voting
    if not multi_comparison:
        pred_class = classes_tested[voted_results(calc_score)]
        pred_df = pd.DataFrame([[configuration_name, class_of_train, pred_class]], columns=['configuration', 'real class', 'prediction'])
        pred_df.to_csv(out_summary_path, mode='a', header=not os.path.exists(out_summary_path), sep=';', index=False)

out_dir = '/Users/miloszwrzesien/Development/cognitiveMaps/new_fcm_module/fcm_results_agg'
main_dir = '/Users/miloszwrzesien/Development/cognitiveMaps/new_fcm_module/fcm_results'
iterate_over_dirs(main_dir, out_dir, use_dtw = False)