from email import header
import os
import json
import pandas as pd

def float_to_num_with_comma(val):
    return str(val).replace('.', ',')

def iterate_over_dirs(main_dir, out_dir, range_dirs: list=None):
    list_dirs = os.listdir(main_dir)
    list_dirs = sorted(list_dirs)
    if range_dirs:
        list_dirs = list_dirs[range_dirs[0]:range_dirs[1]]
    dirs = [os.path.join(main_dir, cd) for cd in list_dirs]
    
    for dir_id, dir in enumerate(dirs):
        conf_names = os.listdir(dir)
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

    out_csv_path = output_path+'.csv'
    out_txt_path = output_path+'_configs.csv'

    # save configuration description
    conf_values = list(con['config'].keys())
    cols_conf = ['conf']+conf_values
    curr_conf_values = [configuration_name] + [str(con['config'][k]) for k in conf_values]
    df = pd.DataFrame([curr_conf_values], columns=cols_conf)
    df.to_csv(out_txt_path, mode='a', header=not os.path.exists(out_txt_path) , sep=';')

    # return 
    cols = ['conf', 'train file(s)', 'train class', 'test file', 'test class', 'mean rmse', 'mean mpe', 'mean max_pe']
    classes_tested = list(con['test results'].keys())
    conf_results = []
    class_of_train = con['files']['class']
    train_files = ','.join(['_'+str(tr) for tr in con['files']['training']])
    for class_n in classes_tested:
        test_files_for_curr_class = list(con['test results'][class_n].keys())
        for test_file in test_files_for_curr_class:
            rmse = con['test results'][class_n][test_file]['rmse'] #todo: reading multiple values (min, mean, max)
            mpe = con['test results'][class_n][test_file]['mpe']
            max_pe = con['test results'][class_n][test_file]['max_pe']
            curr_line = [configuration_name, train_files, class_of_train, test_file, \
                class_n, float_to_num_with_comma(rmse), float_to_num_with_comma(mpe), \
                    float_to_num_with_comma(max_pe)]
            conf_results.append(curr_line)
    conf_df = pd.DataFrame(conf_results, columns=cols)
    conf_df.to_csv(out_csv_path, mode='a', header=not os.path.exists(out_csv_path), sep=';')




out_dir = '/Users/miloszwrzesien/Development/cognitiveMaps/new_fcm_module/fcm_results_agg'
main_dir = '/Users/miloszwrzesien/Development/cognitiveMaps/new_fcm_module/fcm_results'
iterate_over_dirs(main_dir, out_dir)