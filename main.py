import json
import os
import numpy as np
from matplotlib import pyplot as plt
import argparse

from prepare_train import fcm_train
from util import data_prep, metrics, squashing_functions, weight_optimization, window_steps

def main(args):
    # reading passed arguments
    steparg, transform, errorarg, modearg, iterarg, pi, windowarg, amountarg, savepath, dataset, tr_path, te_path,\
    specFiles, specTestFile, rescaleLimits, min_max_sc, class_number, pass_train, ua = args
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
    train_path = f'{dataset}/Train'
    test_path = f'{dataset}/Test'
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
        train_series_set, test_series_set, train_file, test_file = data_prep.import_from_arff(train_path=tr_path, test_path=te_path, amount=amountarg,
                                                                            classif=class_n, dims=3, specificFiles=specFiles,
                                                                            specTestFile=specTestFile, min_max_scale=min_max_sc)


    if ts == "":
        ts = modearg+"_"+errorarg+"_n"+str(amountarg)+"_"+"-".join([tf_name.split(".")[0] for tf_name in train_file])


    # print('tr type', type(train_series_set))
    # print(f'Training files indices: {train_file}')
    # for tr_id, tr_s in enumerate(train_series_set):
    #     print(f'train {tr_id}:', tr_s.max(), tr_s.min())
    # print('test:', test_series_set.max(), test_series_set.min())

    # return 0

    # preparing training ==========================
    weights, errors, loop_error = fcm_train(train_series_set=train_series_set,
                                            step=step, transition_func=transformation,
                                            error=error, mode=mode, max_iter=iterarg,
                                            performance_index=pi,
                                            window=windowarg, use_aggregation=ua,
                                            get_best_weights=True,
                                            passing_files_method=pass_train)
    # =============================================

    return
    if (savepath == None):
        savedist = 'output'
    else:
        savedist = f'{savepath}/output'

    if not os.path.exists(savedist):
        os.makedirs(savedist)
    if not os.path.exists(f'{savedist}/{ts}'):
        os.makedirs(f'{savedist}/{ts}')

    summary = {
        'config': {
            'step': step.__name__,
            'algorithm': 'Nelder-Mead',
            'error': error.__name__,
            'transformation function': transformation.__name__,
            'calculations position': mode.__name__,
            'max iterations': max_iter,
            'window size': window,
            'performance index': performance_index
        },
        'files': {
            'training': train_file,
            'testing': test_file[0],
            'train path': train_path,
            'test path': test_path,
            'class': classif
        },
        'weights': {
            'aggregation': [],
            'fcm': weights.tolist()
        },
        'results': {
            'final error': loop_error,
            'iterations': len(errors),
            'errors': errors
        }
    }

    with open(f"{savedist}/{ts}/train_summary.json", "w") as f:
        json.dump(summary, f)

    f1 = plt.figure(1)
    f1.suptitle('Train errors')
    plt.ylabel(f'{error.__name__}')
    plt.xlabel('outer loop iteration count')
    plt.plot(errors)
    plt.savefig(f'{savedist}/{ts}/train_errors.png', bbox_inches='tight')

    return ts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fuzzy Cognitive Mapping training')
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
    parser.add_argument("-tr_path", dest="tr_path", type=str, help='Path to the .arff train file')
    parser.add_argument("-te_path", dest="te_path", type=str, help='Path to the .arff test file')
    parser.add_argument("-sF", dest="specFiles", type=str, help="Specific training files numbers")
    parser.add_argument("-stF", dest="specTestFile", type=str, help="Specific testing file number")
    parser.add_argument("-limits", dest="rescaleLimits", type=str, help="Rescale limits: min max")
    parser.add_argument("-min_max_sc", dest="min_max_sc", type=str, help="Rescale limits: min max")
    parser.add_argument("-c", dest="c", type=str, help="Perform training on specific class")
    parser.add_argument("-pt", dest="pt", type=str, default='random', help="The method of passing files to training (random, sequential)")
    parser.add_argument("-ua", dest="ua", action='store_true', help="Use aggregation in fcm training")

    args = parser.parse_args()
    argu = args.step, args.transform, args.error, args.mode, args.iter, args.pi, \
            args.window, args.amount, args.savepath, args.dataset, args.tr_path, args.te_path, \
            args.specFiles, args.specTestFile, args.rescaleLimits, args.min_max_sc, args.c, args.pt, args.ua
    main(argu)

