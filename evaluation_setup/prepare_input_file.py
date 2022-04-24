import os
import json
from sklearn.model_selection import ParameterGrid

def list_to_str(lst):
    return str(lst)[1:-1].replace(' ', '')

def get_python_run_commands_from_json(path):
    f = open(path)
    con = json.load(f)
    run_commands = []
    test_commands = [] #todo: add testing commands for specific models (standarized file naming required)
    for conf in con['configs']:
        ds_name = conf['datasets']['name']
        dims = conf['datasets']['dimensions']
        tr_path = conf['datasets']['train_path']
        te_path = conf['datasets']['test_path']

        sF = None
        stF = None
        if type(conf['datasets']['method']) == str:
            if conf['datasets']['method'] == 'random':
                # todo: implement random selection function
                pass
            elif conf['datasets']['method'] == 'crossval':
                # todo: implement crossvalidation function
                pass
        else:
            sF = f"-sF {list_to_str(conf['datasets']['method']['train'])}"
            stF = f"-stF {list_to_str(conf['datasets']['method']['test'])}" if len(conf['datasets']['method']['train']) > 0 else ""
        
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
            run_command = f'python3 main.py -tr_path {tr_path} -te_path {te_path} {sF} {stF} -c {class_n} -cls {cls_type}{cls_params} -dn {ds_name} -dims {dims} --path={output_path}'
            run_commands.append(run_command)
            print(run_command)
            print("")


# todo: function that creates json file for specific configurations
def prepare_python_run_commands():
    configs = []
    cls_params_lists = []
    cls_params = {"-m": ["inner", "outer"], "-e": ["rmse"], "-s": ["distinct", "overlap"], "-i": ["30"]}
    cls_param_grid = list(ParameterGrid(cls_params))
    cls_param_keys = list(cls_params.keys())
    for c_p in cls_param_grid:
        tmp_list = []
        for k in cls_param_keys:
            tmp_list.append([k, c_p[k]])
        cls_params_lists.append(tmp_list)

    param_grid = list(ParameterGrid({
        "class": [0],
        "classifier_params": cls_params_lists
    }))
    for p in param_grid:

        json_dict = {
            "datasets": {
                "name": "UWave",
                "dimensions": 3,
                "train_path": "/Users/miloszwrzesien/Downloads/UWaveGestureLibrary/UWaveGestureLibrary_TRAIN.arff",
                "test_path": "/Users/miloszwrzesien/Downloads/UWaveGestureLibrary/UWaveGestureLibrary_TEST.arff",
                "method": {
                    "train": [0,1],
                    "test": [1]
                }
            },
            "class": p["class"],
            "classifier": "fcm",
            "classifier parameters": p["classifier_params"],
            "output path": "out_test"
        }
        configs.append(json_dict)
    final = dict({"configs": configs})
    with open('data.json', 'w') as fp:
        json.dump(final, fp)

def get_test_run_commands():
    pass
    

prepare_python_run_commands()
get_python_run_commands_from_json('data.json')