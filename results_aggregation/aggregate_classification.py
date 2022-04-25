import os
import json

from numpy import sort

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
            aggregate_from_json(json_path, curr_out_dir)
            break
        break




# todo: IN - json path, out_path, OUT - csv with all results, txt describing configurations
def aggregate_from_json(json_path, output_path):
    with open(json_path, 'r') as f:
        con = json.load(f)

    out_csv_path = output_path+'.csv'

    classes_tested = list(con['test results'].keys())
    print(classes_tested)
    # todo append results to csv


out_dir = '/Users/miloszwrzesien/Development/cognitiveMaps/new_fcm_module/fcm_results/25_04_res'
main_dir = '/Users/miloszwrzesien/Development/cognitiveMaps/new_fcm_module/fcm_results'
iterate_over_dirs(main_dir, out_dir)