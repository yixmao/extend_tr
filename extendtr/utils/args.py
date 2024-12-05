import numpy as np
import argparse
import sys
from extendtr.utils.utils import str2bool

def TopoRegArgs(args=None):
    parser = argparse.ArgumentParser(description="Arguments for Topological Regression (TopoReg)")
    # Anchor selection
    parser.add_argument('-anchorselection', type=str, default='random', 
                        help="Anchor selection strategy: 'random', 'maximin', or 'maximin_density'")
    # Distance settings
    parser.add_argument('-desc_norm', type=str2bool, default=False, help="Normalize the descriptors or not")
    parser.add_argument('-distance', type=str, default='jaccard', 
                        help="Distance metric for calculations. Options: 'jaccard', 'tversky', 'euclidean', 'cosine'")
    # Model configuration
    parser.add_argument('-model', type=str, default='LR', 
                        help="Model type: 'LR', 'LR_L1', 'RF', or 'ANN'")
    # Regular TR settings
    parser.add_argument('-anchor_percentage', type=float, default=0.5, 
                        help="Percentage of anchors for training")
    parser.add_argument('-max_num_anchors', type=int, default=2000, 
                        help="Maximum number of anchor points")
    # Ensemble TR settings
    parser.add_argument('-ensemble', type=str2bool, default=False, help="Enable ensemble Topological Regression (TR)")
    parser.add_argument('-mean_anchor_percentage', type=float, default=0.6, 
                        help="Mean anchor percentage for TR ensemble")
    parser.add_argument('-std_anchor_percentage', type=float, default=0.2, 
                        help="Standard deviation of anchor percentage for TR ensemble")
    parser.add_argument('-min_anchor_percentage', type=float, default=0.3, 
                        help="Minimum anchor percentage for TR ensemble")
    parser.add_argument('-max_anchor_percentage', type=float, default=0.9, 
                        help="Maximum anchor percentage for TR ensemble")
    parser.add_argument('-num_TR_models', type=int, default=15, 
                        help="Number of TR models included in the ensemble")
    
    # General settings
    parser.add_argument('-seed', type=int, default=2021, help="Random seed for reproducibility")
    parser.add_argument('-rbf_gamma', type=float, default=0.5, help="Gamma parameter for the RBF function")
    parser.add_argument('-verbose', type=str2bool, default=False, help="Report the metrics or not")
    parser.add_argument('-check_duplicates', type=str2bool, default=False, help="Check for duplicate samples in the dataset")


    # Tversky distance hyperparameters
    parser.add_argument('-tversky_alpha', type=float, default=0.5, help="Alpha parameter for Tversky distance")
    parser.add_argument('-tversky_beta', type=float, default=0.5, help="Beta parameter for Tversky distance")

    # Lasso hyperparameters
    parser.add_argument('-refine_anchors_lasso', type=str2bool, default=False, 
                        help="Enable L1-norm regularization to refine the anchors")
    parser.add_argument('-lasso_alpha', type=float, default=0.05, help="Alpha parameter for Lasso regularization")
    parser.add_argument('-lasso_thres', type=float, default=1e-6, 
                        help="Threshold for Lasso coefficient filtering")

    # maximin_density hyperparameters
    parser.add_argument('-weight_density', type=float, default=0.5, 
                        help="Weight for the maximin_density anchor selection approach (between 0 and 1)")

    # ANN hyperparameters
    parser.add_argument('-ann_cp_dir', type=str, default='./results/ann_cp/', 
                        help="Directory to save the ANN checkpoint")
    parser.add_argument('-ann_act', type=str, default='tanh', 
                        help="Activation function for ANN: 'tanh', 'relu', 'sigmoid', 'linear'")
    parser.add_argument('-ann_lr', type=float, default=0.001, help="Learning rate for ANN training")
    parser.add_argument('-ann_num_layers', type=int, default=1, help="Number of layers in the ANN")
    parser.add_argument('-ann_epochs', type=int, default=50, help="Number of training epochs for ANN")
    parser.add_argument('-ann_batch_size', type=int, default=256, help="Batch size for ANN training")
    parser.add_argument('-ann_batch_norm', type=str2bool, default=True, 
                        help="Enable batch normalization in ANN")
    parser.add_argument('-ann_init_wts', type=str2bool, default=True, 
                        help="Enable weight initialization in ANN")

    # ANN early stopping
    parser.add_argument('-ann_early_stop', type=str2bool, default=True, help="Enable early stopping for ANN")
    parser.add_argument('-ann_patience', type=int, default=3, 
                        help="Number of steps to wait before stopping ANN training")
    parser.add_argument('-ann_min_delta', type=float, default=1e-3, 
                        help="Minimum change in the NRMSE to qualify as an improvement")

    # Parse the args passed to the function if any
    if args is not None:
        # Simulate the argument passing as if it was from the command line
        sys.argv = [''] + args.split()

    return parser.parse_args()


