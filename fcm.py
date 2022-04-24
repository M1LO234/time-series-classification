import json
import os
import numpy as np
from matplotlib import pyplot as plt
import argparse

from prepare_train import fcm_train
from util import data_prep, metrics, squashing_functions, weight_optimization, window_steps

def fcm(args):
    # reading passed arguments
    cls_type, steparg, transform, errorarg, modearg, iterarg, pi, windowarg, amountarg, savepath, \
    dataset, ds_name, dimensions, tr_path, te_path, specFiles, specTestFile, rescaleLimits, min_max_sc, \
    class_number, pass_train, ua = args

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
                                                                                classif=class_n, dims=dimensions, specificFiles=specFiles,
                                                                                specTestFile=specTestFile, min_max_scale=min_max_sc)

        ts = f'{ds_name}_{cls_type}_{modearg}_{errorarg}_{steparg}_w{windowarg}{"_ua" if ua else ""}_c{class_n}_{"-".join([str(tf) for tf in train_file])}'


        # preparing training ==========================
        weights, errors, loop_error = fcm_train(train_series_set=train_series_set,
                                                step=step, transition_func=transformation,
                                                error=error, mode=mode, max_iter=iterarg,
                                                performance_index=pi,
                                                window=windowarg, use_aggregation=ua,
                                                get_best_weights=True,
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
                'data normalization ranges': min_max_sc
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fuzzy Cognitive Mapping training')
    parser.add_argument('-cls', '--cls_type', dest='cls_type', default="fcm", choices=["fcm", "other"], type=str, help='Classifier type')
    parser.add_argument('-s', '--step', dest='step', default="overlap", choices=["overlap", "distinct"], type=str, help='Steps for training')
    parser.add_argument('-t', '--transformation', dest='transform', default="sigmoid",
     choices=["sigmoid", "binary", "tanh", "arctan", "gaussian"],
     type=str, help='Transformation function')
    parser.add_argument('-e', '--error', dest='error', default="rmse",
     choices=["rmse", "mpe", "max_pe"], type=str, help='Error function')  
    parser.add_argument('-m', '--mode', dest='mode', default="outer",
     choices=["outer", "inner"], type=str, help='Mode of calculations')
    parser.add_argument("-i", "--iter", dest="iter", default=500, type=int, help='Training iterations')
    parser.add_argument("-p", "--performance", dest="pi", default=1e-5, type=float, help='Performance index')
    parser.add_argument("-w", "--window", dest="window", default=4, type=int, help='Size of the window')
    parser.add_argument("-am", "--amount", dest="amount", default=4, type=int, help='Number of training files')
    parser.add_argument("--path", dest="savepath", type=str, help='Path to save the model')
    parser.add_argument("-d", "--dataset", dest="dataset", type=str, help='Path to the dataset')
    parser.add_argument("-dn", "--ds_name", dest="ds_name", default='unknown', type=str, help='Name of the dataset')
    parser.add_argument("-dims", "--dimensions", dest="dimensions", type=str, help='Dimensions of the dataset')
    parser.add_argument("-tr_path", dest="tr_path", type=str, help='Path to the .arff train file')
    parser.add_argument("-te_path", dest="te_path", type=str, help='Path to the .arff test file')
    parser.add_argument("-sF", dest="specFiles", type=str, help="Specific training files numbers")
    parser.add_argument("-stF", dest="specTestFile", type=str, help="Specific testing file number")
    parser.add_argument("-limits", dest="rescaleLimits", type=str, help="Rescale limits: min max")
    parser.add_argument("-min_max_sc", dest="min_max_sc", type=str, help="Rescale limits: min max")
    parser.add_argument("-c", dest="c", type=str, help="Perform training on specific class")
    parser.add_argument("-pt", dest="pt", type=str, default='random', choices=['random', 'sequential'], help="The method of passing files to training (random, sequential)")
    parser.add_argument("-ua", dest="ua", action='store_true', help="Use aggregation in fcm training")

    args = parser.parse_args()
    argu =  args.cls_type, args.step, args.transform, args.error, args.mode, args.iter, args.pi, \
            args.window, args.amount, args.savepath, args.dataset, args.ds_name, args.dimensions, \
            args.tr_path, args.te_path, args.specFiles, args.specTestFile, args.rescaleLimits, \
            args.min_max_sc, args.c, args.pt, args.ua
    fcm(argu)

