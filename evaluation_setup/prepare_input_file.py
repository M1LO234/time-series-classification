import sys
import os
import json
import shutil
import stat
import time
import numpy as np
from pathlib import Path
from sklearn.model_selection import ParameterGrid
from datetime import datetime

def list_to_str(lst):
    return str(lst)[1:-1].replace(' ', '')

def get_files_list_from_json(value):
    if type(value) == str:
        return [i for i in range(int(value.split('-')[0]), int(value.split('-')[1])+1)]
    elif type(value) == list:
        return value

def get_random_ints(length, v_min, v_max):
    out = []
    for l in range(length):
        not_added = True
        while not_added:
            curr_rand = np.random.randint(v_min, v_max)
            if curr_rand not in out:
                out.append(curr_rand)
                not_added = False
    out.sort()
    return out

def parameters_from_range(r):
    range_vals = [int(v) for v in r[1:].split("-")]
    # all integers from range
    if 'a' in r:
        return list(range(range_vals[0], range_vals[1]+1))
    # odd numbers
    if 'o' in r:
        return [v for v in list(range(range_vals[0], range_vals[1]+1)) if v%2 != 0]
    # random integers from interval
    if 'r' in r:
        return get_random_ints(*range_vals)
    # specific values
    if 's' in r:
        return [int(v) for v in r[1:].split('-')]

def get_python_run_commands_from_json(path, flag_params):
    f = open(path)
    con = json.load(f)
    run_commands = []
    test_commands = '' #todo: add testing commands for specific models (standarized file naming required)
    for conf in con['configs']:
        ds_name = conf['datasets']['name']
        dims = conf['datasets']['dimensions']
        tr_path = conf['datasets']['train_path']
        te_path = conf['datasets']['test_path']

        sF, stF, mtf, mtf_cl = None, None, "", ""

        if 'a' in conf['datasets']['files']['test']:
            test_values = parameters_from_range(conf['datasets']['files']['test'])
            conf['datasets']['files']['test'] = test_values

        if conf['datasets']['files']['method'] == 'random':
            train_values = parameters_from_range(conf['datasets']['files']['train'])
            sF = f"-sF {list_to_str(train_values)}"
            stF = f"-stF {list_to_str(conf['datasets']['files']['test'])}" if len(conf['datasets']['files']['test']) == 0 else f"-stF {list_to_str([1])}"
            mtf = f"-mtf {list_to_str(get_files_list_from_json(conf['datasets']['files']['test']))}"
            mtf_cl = f"-mtf_class {list_to_str(conf['datasets']['files']['test_classes'])}"

        elif conf['datasets']['files']['method'] == 'crossval':
            # todo: implement crossvalidation function
            pass
        elif conf['datasets']['files']['method'] == 'specific':
            sF = f"-sF {list_to_str(get_files_list_from_json(conf['datasets']['files']['train']))}"
            stF = f"-stF {list_to_str(conf['datasets']['files']['test'])}" if len(conf['datasets']['files']['test']) == 0 else f"-stF {list_to_str([1])}"
            mtf = f"-mtf {list_to_str(get_files_list_from_json(conf['datasets']['files']['test']))}"
            mtf_cl = f"-mtf_class {list_to_str(conf['datasets']['files']['test_classes'])}"

        class_n = conf['class']
        cls_type = conf['classifier']
        cls_params = ''
        for p_val in conf['classifier parameters']:
            if len(p_val) == 1:
                cls_params += f' {p_val[0]}'
            elif len(p_val) == 2:
                cls_params += f' {p_val[0]} {p_val[1]}'

        output_path = conf['output path']

        if cls_type == 'fcm':
            run_command = f'python3 main.py -tr_path {tr_path} -te_path {te_path} {sF} {stF} {" ".join(flag_params)+" " if len(flag_params) > 0 else ""}-c {class_n} -cls {cls_type}{cls_params} -dn {ds_name} -dims {dims} --path={output_path} {mtf} {mtf_cl}'
            run_commands.append(run_command)
            run_f_name = f'{conf["classifier"]}_{conf["class"]}_{int(time.time())}.sh'
        elif cls_type == "":
            # TODO: add commands for another classifiers
            pass

        with open(run_f_name, 'w') as fp:
            final_com = " & ".join(run_commands) + ' & wait'
            fp.write(final_com)

        st = os.stat(run_f_name)
        os.chmod(run_f_name, st.st_mode | stat.S_IEXEC)

    abs_f_path = os.path.join(os.getcwd(), run_f_name)
    parent_dir = Path(abs_f_path).parents[1]
    shutil.move(abs_f_path, parent_dir)
            
    # print(len(run_commands))
    return run_f_name

def get_test_run_commands():
    pass

def get_input_from_user(json_path, out_file_name):
    with open(json_path, 'r') as f:
        con = json.load(f)

    commands = ''
    for c in con["classes"]:
        configs, cls_params_lists = [], []
        cls_params = dict([(key, value) for key, value in con['cls_params'].items() if key not in ['-flag_params']])
        flag_params = con['cls_params']['-flag_params'] if '-flag_params' in list(con['cls_params']) else []
        cls_param_grid = list(ParameterGrid(cls_params))
        cls_param_keys = list(cls_params.keys())
        for c_p in cls_param_grid:
            tmp_list = []
            for k in cls_param_keys:
                tmp_list.append([k, c_p[k]])
            cls_params_lists.append(tmp_list)

        param_grid = list(ParameterGrid({
            "class": [c],
            "classifier_params": cls_params_lists,
            "train": con["files"]["train"]
        }))
        day_month = datetime.now().strftime("%d_%m_%H_%M")

        for conf_id, p in enumerate(param_grid):
            json_dict = {
                "datasets": {
                    "name": con["name"],
                    "dimensions": con["dimensions"],
                    "train_path": con['train_path'],
                    "test_path": con['test_path'],
                    "files": {
                        "method": con['files']['method'],
                        "train": p["train"],
                        "train_samples": con['files']['train_samples'],
                        "test": con['files']['test'],
                        "test_classes": con['files']['test_classes']
                    }
                },
                "class": p["class"],
                "classifier": con['classifier'],
                "classifier parameters": p["classifier_params"],
                "output path": f"{day_month}_{con['name']}_class{p['class']}/{day_month}_{con['name']}_class{p['class']}_conf{conf_id}"
            }
            configs.append(json_dict)
        final = dict({"configs": configs})
        out_name = f'data{c}.json'
        with open(out_name, 'w') as fp:
            json.dump(final, fp)
        file_name = get_python_run_commands_from_json(out_name, flag_params)
        commands += f'./{file_name} & \n'
    commands += ' wait'

    # todo: add eden script
    with open(f'{out_file_name}.sh', 'w') as fp:
        fp.write(commands)
    st = os.stat(f'{out_file_name}.sh')
    os.chmod(f'{out_file_name}.sh', st.st_mode | stat.S_IEXEC)
    abs_f_path = os.path.join(os.getcwd(), f'{out_file_name}.sh')
    parent_dir = Path(abs_f_path).parents[1]
    shutil.move(abs_f_path, parent_dir)

if __name__ == '__main__':
    args = sys.argv[1:]
    get_input_from_user(args[0], args[1])