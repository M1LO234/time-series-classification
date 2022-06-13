import json
import os
import numpy as np
from prepare_train import fcm_train
from util.fcm_util import data_prep, metrics, squashing_functions, weight_optimization, window_steps

def fcm(args):
    # reading passed arguments
    cls_type, steparg, transform, errorarg, modearg, iterarg, pi, windowarg, amountarg, savepath, \
    dataset, ds_name, dimensions, tr_path, te_path, specFiles, specTestFile, rescaleLimits, min_max_sc, \
    class_number, pass_train, ua, best_weights = args

    if cls_type == "fcm":
        if specFiles: #TODO standardize sepcFiles format
            if "," in specFiles: # create list of multiple training files
                list_spec_files = specFiles.split(",")
                ts = modearg+"_"+errorarg+"_n"+str(amountarg)+"_"+ "-".join(sp_f for sp_f in list_spec_files) # output path
                if not tr_path:
                    specFiles = [f_name+".csv" for f_name in specFiles.split(",")]
                else:
                    specFiles = [f_name for f_name in specFiles.split(",")]
            else: # one training file
                ts = modearg+"_"+errorarg+"_n"+str(amountarg)+"_"+ specFiles
                if not tr_path:
                    specFiles = specFiles+".csv"
        else:
            ts = ""
        if rescaleLimits: # scaling fith custom values
            rescaleLimits = [[float(val) for val in lim.split(',')] for lim in rescaleLimits.split(";")]
        if (steparg == "overlap"):
            step = window_steps.overlap_steps
        else:
            step = window_steps.distinct_steps
        if (transform == "sigmoid"):
            transformation = squashing_functions.sigmoid
        elif (transform == "binary"):
            transformation = squashing_functions.binary
        elif (transform == "tanh"):
            transformation = squashing_functions.tanh
        elif (transform == "arctan"):
            transformation = squashing_functions.arctan
        else:
            transformation = squashing_functions.gaussian
        if (errorarg == "rmse"):
            error = metrics.rmse
        elif (errorarg == "mpe"):
            error = metrics.mpe
        else:
            error = metrics.max_pe
        if (modearg == "outer"):
            mode = weight_optimization.outer
        else:
            mode = weight_optimization.inner

        if dataset:
            train_path = f'{dataset}/Train'
            test_path = f'{dataset}/Test'
        else:
            train_path = tr_path
            test_path = te_path

        if dimensions:
            dimensions = int(dimensions)
        else:
            dimensions = 1

        if class_number:
            class_n = int(class_number)
        else:
            class_n = 1
        if min_max_sc:
            min_max_sc = [float(v) for v in min_max_sc.split(',')]

        if not tr_path:
            train_series_set, test_series_set, train_file, test_file = data_prep.import_from_dataset(amountarg, train_path=train_path,
                                                                                        test_path=test_path, classif=class_n, 
                                                                                        specificFiles=specFiles,
                                                                                        specTestFile=specTestFile,
                                                                                        min_max_scale=min_max_sc,
                                                                                        rescLimits=rescaleLimits)
        else:
            train_series_set, test_series_set, train_file, test_file = data_prep.import_from_arff(train_path=train_path, test_path=test_path,
                                                                                class_train=class_n, dims=dimensions, specificFiles=specFiles,
                                                                                specTestFile=specTestFile, min_max_scale=min_max_sc)

        ts = f'{ds_name}_{cls_type}_{modearg}_{errorarg}_{steparg}_w{windowarg}{"_ua" if ua else ""}_c{class_n}_{"-".join([str(tf) for tf in train_file])}'

        # preparing training ==========================
        weights, errors, loop_error = fcm_train(train_series_set=train_series_set,
                                                step=step, transition_func=transformation,
                                                error=error, mode=mode, max_iter=iterarg,
                                                performance_index=pi,
                                                window=windowarg, use_aggregation=ua,
                                                get_best_weights=best_weights,
                                                passing_files_method=pass_train)

        agg_weights = weights[1]
        weights = weights[0]
        # =============================================

        # return
        if (savepath == None):
            savedist = 'output'
        else:
            savedist = f'{savepath}'

        if not os.path.exists(savedist):
            os.makedirs(savedist)

        summary = {
            'config': {
                'step': step.__name__,
                'algorithm': 'Nelder-Mead',
                'error': error.__name__,
                'transformation function': transformation.__name__,
                'calculations position': mode.__name__,
                'max iterations': iterarg,
                'window size': windowarg,
                'performance index': pi,
                'data normalization ranges': min_max_sc,
                'passing files method': pass_train,
                'best weights': best_weights,
                'aggregation used': True if type(agg_weights) == np.ndarray else False
            },
            'files': {
                'training': train_file,
                'testing': test_file[0],
                'train path': train_path,
                'test path': test_path,
                'class': class_n,
                'dimensions': dimensions
            },
            'weights': {
                'aggregation': agg_weights.tolist() if type(agg_weights) == np.ndarray else None,
                'fcm': weights.tolist()
            },
            "train results": {},
            "test results": {},
            'results': {
                'final error': loop_error,
                'iterations': len(errors),
                'errors': errors
            }
        }

        with open(f"{savedist}/{ts}.json", "w") as f:
            json.dump(summary, f)

        return os.path.join(savedist, ts+'.json')