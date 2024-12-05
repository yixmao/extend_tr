import time
import numpy as np
import pandas as pd
import os
import warnings
# 
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.exceptions import ConvergenceWarning
#
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
#
from extendtr.utils.utils import rbf, check_and_remove_duplicates, normalize_data
from extendtr.utils.ann_model import ANN_Model, train_model
from extendtr.TR.distance import calculate_distances
from extendtr.TR.anchor_selection import select_anchor



def TopoReg(desc, target, train_idx, test_idx, val_idx, args):
    """
    Implements the Topological Regression (TopoReg) pipeline, including optional ensemble and stacking variants.

    Parameters:
    - desc: pd.DataFrame, descriptors
    - target: pd.Series or pd.DataFrame, target values
    - train_idx: list, indices for training samples
    - test_idx: list, indices for testing samples
    - val_idx: list, indices for validation samples (needed for ANN model and stacking over configurations, otherwise not used)
    - args: Namespace, contains arguments

    Returns:
    - mdl: trained model(s) (varies by approach)
    - pred_test: np.ndarray, predictions for the test set
    - pred_val: np.ndarray, predictions for the validation set (if applicable)
    - train_time: float, total training time
    - test_time: float, total testing time
    """

    # normalize the descriptors
    if args.desc_norm:
        idx = desc.index
        desc, _, _ = normalize_data(desc.values)
        desc = pd.DataFrame(desc, index=idx)
    # Calculate distances
    distance_x = calculate_distances(desc.values, args)
    distance_x = pd.DataFrame(distance_x, index=desc.index, columns=desc.index) 
    distance_y = pairwise_distances(target.values.reshape(-1, 1), metric="euclidean", n_jobs=-1)
    distance_y = pd.DataFrame(distance_y, index=target.index, columns=target.index)
    # check and remove duplicates in the training samples
    if args.check_duplicates:
        train_idx = check_and_remove_duplicates(distance_x, train_idx, args.verbose)
    # 
    if args.ensemble: # ensemble TR
        mdl, pred_test, pred_val, train_time, test_time = ensemble_TR(distance_x, distance_y, target, train_idx, test_idx, val_idx, args)
    else: # normal TR
        anchor_percentage = args.anchor_percentage
        # Sample anchor points
        anchors_idx = select_anchor(distance_x, distance_y, train_idx, anchor_percentage, args)
        # model and predict
        mdl, pred_test, pred_val, train_time, test_time = mdl_pred(distance_x, distance_y, target, train_idx, anchors_idx, test_idx, val_idx, args) 
        
    return mdl, pred_test, pred_val, train_time, test_time


def ensemble_TR(distance_x, distance_y, target, train_idx, test_idx, val_idx, args):
    """
    Performs ensemble-based Topological Regression by averaging predictions from multiple models
    trained with varying anchor percentages and radnomly selected anchors.

    Parameters:
    - distance_x: pd.DataFrame, structure (input) distance matrix
    - distance_y: pd.DataFrame, response distance matrix
    - target: pd.Series or pd.DataFrame, target values
    - train_idx: list, training sample indices
    - test_idx: list, test sample indices
    - val_idx: list, validation sample indices (optional)
    - args: Namespace, contains arguments for ensemble configuration

    Returns:
    - models: list, trained models
    - pred_test: np.ndarray, ensemble-averaged predictions for the test set
    - pred_val: np.ndarray, ensemble-averaged predictions for the validation set (if applicable)
    - total_train_time: float, total training time
    - total_test_time: float, total testing time
    """
    # Sample anchor point percentages
    # if args.anchorselection == 'random':
    anchor_percentages = np.random.normal(args.mean_anchor_percentage,args.std_anchor_percentage,args.num_TR_models)
    # Clip anchor percentages to assert validity and prevent under/overfitting
    anchor_percentages[anchor_percentages<args.min_anchor_percentage]=args.min_anchor_percentage
    anchor_percentages[anchor_percentages>args.max_anchor_percentage]=args.max_anchor_percentage
    # else: 
        # anchor_percentages = args.anchor_percentage_options
    # Perform modeling+prediction for each achor point percentage
    preds_test = []
    preds_val = []
    models = []
    total_train_time = 0
    total_test_time = 0
    for anchor_percentage in anchor_percentages:
        # Sample anchor points
        anchors_idx = select_anchor(distance_x, distance_y, train_idx, anchor_percentage, args)
        # model and predict
        mdl, pred_test, pred_val, train_time, test_time = mdl_pred(distance_x, distance_y,target, train_idx, anchors_idx, test_idx, val_idx, args)
        # Accumulate the training and testing times
        total_train_time += train_time
        total_test_time += test_time
        # 
        preds_test.append(pred_test)
        preds_val.append(pred_val)
        models.append(mdl)
    # Ensemble average   
    pred_test = np.array(preds_test).mean(axis=0)
    if val_idx is not None:
        pred_val = np.array(preds_val).mean(axis=0)
    else:
        pred_val = None
    return models, pred_test, pred_val, total_train_time, total_test_time


def mdl_pred(distance_x, distance_y, target, train_idx, anchors_idx, test_idx, val_idx, args):
    """
    Trains a model using the selected anchor points and makes predictions for the test and validation sets.

    Parameters:
    - distance_x: pd.DataFrame, structure (input) distance matrix
    - distance_y: pd.DataFrame, response distance matrix
    - target: pd.Series or pd.DataFrame, target values
    - train_idx: list, indices of training samples
    - anchors_idx: list, indices of selected anchor points
    - test_idx: list, indices of test samples
    - val_idx: list, indices of validation samples (optional, can be None)
    - args: Namespace, contains model configuration (e.g., model type, ANN hyperparameters)

    Returns:
    - mdl: trained model
    - pred_test: np.ndarray, predictions for the test set
    - pred_val: np.ndarray, predictions for the validation set (if applicable)
    - train_time: float, training time
    - test_time: float, testing time
    """
    # Modelling - start training
    if args.model in ['LR', 'LR_L1', 'RF']:
        # Sample training and testing distances
        if val_idx is None:
            pred_val = None 
        else:
            dist_x_val = distance_x.loc[val_idx, anchors_idx]
        dist_x_train = distance_x.loc[train_idx, anchors_idx]
        dist_y_train = distance_y.loc[train_idx, anchors_idx]
        dist_test = distance_x.loc[test_idx, anchors_idx]
        # start training
        start_train_time = time.time() 
        if args.model == 'LR':
            mdl = LR(n_jobs=-1)
        elif args.model == 'LR_L1':
            mdl = Lasso(alpha=args.lasso_alpha)
        elif args.model == 'RF':
            mdl = RF(n_jobs=-1)

        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        mdl.fit(dist_x_train, dist_y_train)
        # make predictions for validation set
        if val_idx is not None:
            dist_array_val = abs(mdl.predict(dist_x_val)).T
            # print(dist_array_test)
            pred_val = rbf(dist_array_val, target, anchors_idx, args.rbf_gamma).ravel()
        # end training
        train_time = time.time() - start_train_time

        # Prediction
        start_test_time = time.time()
        dist_array_test = abs(mdl.predict(dist_test)).T
        # print(dist_array_test)
        pred_test = rbf(dist_array_test, target, anchors_idx, args.rbf_gamma).ravel()
        # End testing
        test_time = time.time() - start_test_time

    elif args.model == 'ANN':
        if val_idx is None:
            raise ValueError("ANN needs validation set.")
        # 
        X = distance_x.loc[:, anchors_idx]
        Y = distance_y.loc[:, anchors_idx]
        # Hyperparameters
        input_size = X.shape[1]
        hidden_size = X.shape[1]
        output_size = Y.shape[1]
        learning_rate = args.ann_lr
        num_epochs = args.ann_epochs
        batch_size = args.ann_batch_size

        # normalize the input
        idx = X.index
        X, _, _ = normalize_data(X.values)
        X = pd.DataFrame(X, index=idx)
        # Convert data to PyTorch tensors
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_train = torch.tensor(X.loc[train_idx].values, dtype=torch.float32).to(device)
        Y_train = torch.tensor(Y.loc[train_idx].values, dtype=torch.float32).to(device)
        X_val = torch.tensor(X.loc[val_idx].values, dtype=torch.float32).to(device)
        Y_val = torch.tensor(Y.loc[val_idx].values, dtype=torch.float32).to(device)
        X_test = torch.tensor(X.loc[test_idx].values, dtype=torch.float32).to(device)
        # Create DataLoader for training
        train_dataset = TensorDataset(X_train, Y_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size)

        # Instantiate the model, define the loss function and the optimizer
        mdl = ANN_Model(input_size, hidden_size, output_size, args).to(device)
        loss_func = nn.MSELoss()
        optimizer = optim.Adam(mdl.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.95)
        # Training the model
        start_train_time = time.time()
        checkpoint_dir = args.ann_cp_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        train_model(mdl, train_loader, val_loader, loss_func, optimizer, num_epochs, args, scheduler, checkpoint_dir)
        # make predictions for validation set
        if val_idx is not None:
            mdl.eval()
            with torch.no_grad():
                dist_array_val = abs(mdl(X_val).cpu().numpy().T)
            pred_val = rbf(dist_array_val, target, anchors_idx, args.rbf_gamma).ravel()
        train_time = time.time() - start_train_time

        # Prediction
        start_test_time = time.time()
        mdl.eval()
        with torch.no_grad():
            dist_array_test = abs(mdl(X_test).cpu().numpy().T)
        # 
        pred_test = rbf(dist_array_test, target, anchors_idx, args.rbf_gamma).ravel()
        # End testing
        test_time = time.time() - start_test_time

    else:
        raise ValueError("Unsupported model type. Choose 'LR', 'LR_L1', 'RF', and 'ANN'.")
    return mdl, pred_test, pred_val, train_time, test_time
    
