"""
This code performs TR with various configurations
"""

import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split

from extendtr.TR.topoReg import TopoReg
from extendtr.utils.args import TopoRegArgs
from extendtr.utils.utils import set_seed, metric_calc, stack_models


if __name__ == "__main__":
    # load the descriptor - the indices of desc will be used later
    desc = pd.read_parquet(f'./SampleDatasets/CHEMBL278/data_ECFP4.parquet', engine='fastparquet').astype('bool')
    # load targets
    data = pd.read_csv(f'./SampleDatasets/CHEMBL278/data_cp.csv', index_col=0)
    target = data["pChEMBL Value"]
    # make sure that the indices of desc and target match
    desc = desc.loc[target.index]
    target = target.loc[desc.index]

    # load indicies for scaffold split
    with open(f'./SampleDatasets/CHEMBL278/scaffold_split_index.json', 'r') as f:
        index = json.load(f)  
    train_idx = index['train_idx']
    test_idx = index['test_idx']
    # make sure that train and test indices are included in target.index
    train_idx = [idx for idx in train_idx if idx in target.index]
    test_idx = [idx for idx in test_idx if idx in target.index]

    ##### alternatively, you can ranomly split train and test idx
    # dataset_idx = target.index.tolist()
    # train_idx, test_idx = train_test_split(dataset_idx, test_size=0.2, random_state=args.seed)

    # set validation index if necessary
    val_set = 0.2 # a fraction number to use [val_set] percent samples from the train set as val set, or None for no validation
    if val_set is not None: # if we want to test on the validation set
        train_idx, val_idx = train_test_split(train_idx, test_size=val_set, random_state=2021)
    else: # no validation
        val_idx = None


    print('----------------- Running different descriptors -----------------')
    # define arg strs for different anchor selection methods
    args = TopoRegArgs(f'-ensemble 1')
    descriptors = ['ECFP4', 'ECFP6', 'Mordred', 'RDKdesc', 'tcnn_zpad']
    data_path = './SampleDatasets/CHEMBL278/'

    ########################### Train and get the prediction ###############################
    preds_test = []
    preds_val = []
    for descriptor in descriptors:
        # Load descriptors
        if descriptor in ['ECFP4', 'ECFP6']:
            desc = pd.read_parquet(f'{data_path}/data_{descriptor}.parquet', engine='fastparquet').astype('bool')
        elif descriptor in ['Mordred', 'RDKdesc']:
            desc = pd.read_parquet(f'{data_path}/data_{descriptor}.parquet', engine='fastparquet')
        elif descriptor == 'tcnn_zpad':
            desc = pd.read_csv(f"{data_path}/tcnn_embeddings_zpad.csv", index_col=0)
        
        if descriptor in ['ECFP4', 'ECFP6']:
            args.distance = 'jaccard'
            args.desc_norm = False
        else:
            args.distance = 'euclidean'
            args.desc_norm = True

        # set random seed
        set_seed(args.seed)
        mdl, pred_test, pred_val, train_time, test_time = TopoReg(desc, target, train_idx, test_idx, val_idx, args)
        # Stack the predicted responses for test and validation sets
        preds_test.append(pred_test)
        preds_val.append(pred_val)

    ########################### Combine the results ###############################
    # ensemble
    pred_test = np.array(preds_test).mean(axis=0)
    # evaluation
    print('Performance of ensemble predictions:')
    scorr, r2, rmse, nrmse = metric_calc(pred_test, target.loc[test_idx], True)

    # stacking the results
    pred_test, train_time, test_time = stack_models(preds_val, preds_test, target, val_idx)
    # evaluation
    print('Performance of stacking predictions:')
    scorr, r2, rmse, nrmse = metric_calc(pred_test, target.loc[test_idx], True)


    print('----------------- Running different anchor selections -----------------')
    # load the descriptor - the indices of desc will be used later
    desc = pd.read_parquet(f'./SampleDatasets/CHEMBL278/data_ECFP4.parquet', engine='fastparquet').astype('bool')
    # define arg strs for different anchor selection methods
    arg_strs_anchor = [f'-ensemble 1', 
                    f'-anchorselection maximin -ensemble 1 -mean_anchor_percentage 0.4 -min_anchor_percentage 0.2 -max_anchor_percentage 0.6',
                    f'-refine_anchors_lasso 1 -anchor_percentage 0.8', 
                    f'-anchorselection maximin_density -weight_density 0.5 -check_duplicates 1'
                    ]


    ########################### Train and get the prediction ###############################
    preds_test = []
    preds_val = []
    for arg_str in arg_strs_anchor:
        args = TopoRegArgs(f'{arg_str}')
        # set random seed
        set_seed(args.seed)
        mdl, pred_test, pred_val, train_time, test_time = TopoReg(desc, target, train_idx, test_idx, val_idx, args)
        # Stack the predicted responses for test and validation sets
        preds_test.append(pred_test)
        preds_val.append(pred_val)

    ########################### Combine the results ###############################
    # ensemble
    pred_test = np.array(preds_test).mean(axis=0)
    # evaluation
    print('Performance of ensemble predictions:')
    scorr, r2, rmse, nrmse = metric_calc(pred_test, target.loc[test_idx], True)


    # stacking the results
    pred_test, train_time, test_time = stack_models(preds_val, preds_test, target, val_idx)
    # evaluation
    print('Performance of stacking predictions:')
    scorr, r2, rmse, nrmse = metric_calc(pred_test, target.loc[test_idx], True)


    print('----------------- Running different distance calculations -----------------')
    # load the descriptor - the indices of desc will be used later
    desc = pd.read_parquet(f'./SampleDatasets/CHEMBL278/data_ECFP4.parquet', engine='fastparquet').astype('bool')
    # define arg strs for different anchor selection methods
    arg_strs_dist = [f'-ensemble 1', 
                    f'-distance tversky -ensemble 1',
                    f'-distance euclidean -ensemble 1', 
                    f'-distance cosine -ensemble 1'
                    ]


    ########################### Train and get the prediction ###############################
    preds_test = []
    preds_val = []
    for arg_str in arg_strs_dist:
        args = TopoRegArgs(f'{arg_str}')
        # set random seed
        set_seed(args.seed)
        mdl, pred_test, pred_val, train_time, test_time = TopoReg(desc, target, train_idx, test_idx, val_idx, args)
        # Stack the predicted responses for test and validation sets
        preds_test.append(pred_test)
        preds_val.append(pred_val)

    ########################### Combine the results ###############################
    # ensemble
    pred_test = np.array(preds_test).mean(axis=0)
    # evaluation
    print('Performance of ensemble predictions:')
    scorr, r2, rmse, nrmse = metric_calc(pred_test, target.loc[test_idx], True)


    # stacking the results
    pred_test, train_time, test_time = stack_models(preds_val, preds_test, target, val_idx)
    # evaluation
    print('Performance of stacking predictions:')
    scorr, r2, rmse, nrmse = metric_calc(pred_test, target.loc[test_idx], True)

    print('----------------- Running different models -----------------')
    # load the descriptor - the indices of desc will be used later
    desc = pd.read_parquet(f'./SampleDatasets/CHEMBL278/data_ECFP4.parquet', engine='fastparquet').astype('bool')
    # define arg strs for different anchor selection methods
    arg_strs_mdls = [f'-ensemble 1', 
                    f'-model LR_L1 -ensemble 1',
                    f'-model RF -ensemble 1', 
                    f'-model ANN -ensemble 1'
                    ]


    ########################### Train and get the prediction ###############################
    preds_test = []
    preds_val = []
    for arg_str in arg_strs_dist:
        args = TopoRegArgs(f'{arg_str}')
        # set random seed
        set_seed(args.seed)
        mdl, pred_test, pred_val, train_time, test_time = TopoReg(desc, target, train_idx, test_idx, val_idx, args)
        # Stack the predicted responses for test and validation sets
        preds_test.append(pred_test)
        preds_val.append(pred_val)

    ########################### Combine the results ###############################
    # ensemble
    pred_test = np.array(preds_test).mean(axis=0)
    # evaluation
    print('Performance of ensemble predictions:')
    scorr, r2, rmse, nrmse = metric_calc(pred_test, target.loc[test_idx], True)


    # stacking the results
    pred_test, train_time, test_time = stack_models(preds_val, preds_test, target, val_idx)
    # evaluation
    print('Performance of stacking predictions:')
    scorr, r2, rmse, nrmse = metric_calc(pred_test, target.loc[test_idx], True)

    print('----------------- All configurations run successfully! -----------------')