import json
import os
import argparse

from fcm import fcm
from testing import test_fcm

def list_from_str(string):
    return string.split(',')

def main(args, mtf):
    res_file = fcm(args)
    # todo: mulitple test files
    if mtf:
        for tf in list_from_str(mtf):
            test_fcm(res_file, test_file=tf)
    else:
        test_fcm(res_file)

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
    parser.add_argument("-mtf", dest="testing_files", help="Multiple testing files")

    args = parser.parse_args()
    argu =  args.cls_type, args.step, args.transform, args.error, args.mode, args.iter, args.pi, \
            args.window, args.amount, args.savepath, args.dataset, args.ds_name, args.dimensions, \
            args.tr_path, args.te_path, args.specFiles, args.specTestFile, args.rescaleLimits, \
            args.min_max_sc, args.c, args.pt, args.ua
    main(argu, args.testing_files)

