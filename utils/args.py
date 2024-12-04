import numpy as np
import argparse
import sys
from utils.utils import str2bool

def TopoRegArgs(args=None):
    parser = argparse.ArgumentParser()
    # parser.add_argument('-path', type=str, default='data/ChEMBL/batch1/CHEMBL1900')
    # parser.add_argument('-results_path', type=str, default='./results/')
    # parser.add_argument('-split', type=str, default='scaffold') # scaffold, random, or cv
    parser.add_argument('-seed', type=int, default=2021)
    # parser.add_argument('-cv_fold', type=int, default=5)
    # parser.add_argument('-test_perc', type=float, default=0.2) # percent of the test samples
    parser.add_argument('-rbf_gamma', type=float, default=0.5)
    # report the metrics or not
    parser.add_argument('-verbose', type=str2bool, default=False)
    # 
    parser.add_argument('-check_duplicates', type=str2bool, default=False)
    # 
    parser.add_argument('-ensemble', type=str2bool, default=False) # ensemble TR
    parser.add_argument('-stack', type=str2bool, default=False) # stack TR
    parser.add_argument('-anchorselection', type=str, default='random') # 'random', 'maximin', 'maximin_density'
    # discriptor and distances
    parser.add_argument('-descriptor', type=str, default='ECFP4') # 'E3FP', 'ECFP4', 'ECFP6', 'Mordred', 'RDKdesc', 'tcnn_avg', 'tcnn_zpad'
    parser.add_argument('-desc_norm', type=str2bool, default=False) # normalize the descriptor or not
    parser.add_argument('-distance', type=str, default='jaccard') # for ['ECFP4', 'ECFP6']: 'jaccard', 'tversky'
                                                               # for others: 'euclidean', 'cosine'
    # models
    parser.add_argument('-model', type=str, default='LR') # LR, LR_L1, RF, and ANN
    # integrate over methods
    # parser.add_argument('-integrate', type=str, default='stack') # stack, ensemble
    parser.add_argument('-val_set', type=float, default=None) # input a fraction number to use [val_set] percent samples from the train set as val set
    # for normal TR
    parser.add_argument('-anchor_percentage', type=float, default=0.5)
    parser.add_argument('-max_num_anchors', type=int, default=2000) # max number of anchor points
    # for ensemble TR
    parser.add_argument('-mean_anchor_percentage', type=float, default=0.6)   # mean anchor point percentage for TR ensemble
    parser.add_argument('-std_anchor_percentage', type=float, default=0.2)    # std of anchor point percentage for TR ensemble
    parser.add_argument('-min_anchor_percentage', type=float, default=0.3)    # min of anchor point percentage for TR ensemble
    parser.add_argument('-max_anchor_percentage', type=float, default=0.9)    # max of anchor point percentage for TR ensemble
    parser.add_argument('-num_TR_models', type=int, default= 30 )  # Number of TR models included in ensemble
    # for tversky distance
    parser.add_argument('-tversky_alpha', type=float, default=0.5)
    parser.add_argument('-tversky_beta', type=float, default=0.5)
    # l1-norm regularization to refine the anchors
    parser.add_argument('-refine_anchors_lasso', type=str2bool, default=False)
    parser.add_argument('-lasso_alpha', type=float, default=0.01)
    parser.add_argument('-lasso_thres', type=float, default=1e-6)
    # weight for the density-based min-max approach
    parser.add_argument('-weight_density', type=float, default=0.5) # between 0 and 1: higher value represents more weight to the density
    # hyperparameters for ANN
    parser.add_argument('-ann_cp_dir', type=str, default='./results/ann_cp/') # directory to save the ANN checkpoint
    parser.add_argument('-ann_act', type=str, default='tanh') # tanh, relu, sigmoid, linaer
    parser.add_argument('-ann_lr', type=float, default='0.001') # learning rate
    parser.add_argument('-ann_num_layers', type=int, default=1) # num layers
    parser.add_argument('-ann_epochs', type=int, default='50') # number of epochs
    parser.add_argument('-ann_batch_size', type=int, default='256') # batch size
    parser.add_argument('-ann_batch_norm', type=str2bool, default=True) # batch normalization
    parser.add_argument('-ann_init_wts', type=str2bool, default=True) # weight initialization
    # early stop for ann
    parser.add_argument('-ann_early_stop', type=str2bool, default=True)
    parser.add_argument('-ann_patience', type=int, default=3) # Number of steps to wait before stopping
    parser.add_argument('-ann_min_delta', type=float, default=1e-5) # Minimum change in the NRMSE to qualify as an improvement

    # Parse the args passed to the function if any
    if args is not None:
        # Simulate the argument passing as if it was from the command line
        sys.argv = [''] + args.split()

    return parser.parse_args()


