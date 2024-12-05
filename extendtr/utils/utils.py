import argparse
import time
import numpy as np
import torch
import random
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression as LR

def set_seed(seed):
    """
    Sets the random seed for reproducibility across various libraries.

    Parameters:
    - seed: int, the seed value to set.

    Effect:
    - Sets the seed for Python's `random`, NumPy, and PyTorch to ensure consistent results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def str2bool(v):
    """
    Converts a string representation of a boolean to an actual boolean value.

    Parameters:
    - v: str, input string (e.g., 'yes', 'true', '1', 'no', 'false', '0').

    Returns:
    - bool.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
# Calculate and stack metrics
def metric_calc(predictedResponse, target, verbose=False):
    """
    Calculates evaluation metrics.

    Parameters:
    - predictedResponse: np.ndarray, predicted responses.
    - target: np.ndarray, actual target values.
    - verbose: bool, report the metrics or not.

    Returns:
    - scorr: float, Spearman correlation coefficient.
    - r2: float, R-squared value.
    - rmse: float, Root Mean Square Error.
    - nrmse: float, Normalized RMSE (normalized by target standard deviation).

    Prints metrics if `args.verbose` is True.
    """
    scorr, pvalue = spearmanr(target, predictedResponse)
    r2 = r2_score(target, predictedResponse)
    rmse = mean_squared_error(target, predictedResponse, squared=False)
    std_i = np.std(target)
    nrmse = rmse / std_i
    if verbose:
        print('Spearman: '+str(scorr))
        print('R2: '+str(r2))
        print('RMSE: '+str(rmse))
        print('NRMSE: '+str(nrmse))
    return scorr, r2, rmse, nrmse


def stack_models(preds_val, preds_test, target, val_idx):
    """
    Combines predictions from multiple configurations using a stacking approach with a linear regression model.

    Parameters:
    - preds_val: list, predictions on the validation set from multiple models.
    - preds_test: list, predictions on the test set from multiple models.
    - target: pd.Series, actual target values.
    - val_idx: list, indices of the validation set.

    Returns:
    - pred_test: np.ndarray, stacked predictions for the test set.
    - train_time: float, training time for the stacking model.
    - test_time: float, testing time for the stacking model.
    """
    start_train_time = time.time() 
    preds_val_np = np.array(preds_val).T
    # Create and fit the linear model on the validation predictions
    stacking_model = LR(n_jobs=-1)
    stacking_model.fit(preds_val_np, target.loc[val_idx].values)
    train_time = time.time() - start_train_time

    # Use the stacking model to combine predictions on the test set
    start_test_time = time.time()
    preds_test_np = np.array(preds_test).T
    pred_test = stacking_model.predict(preds_test_np)
    test_time = time.time() - start_test_time
    return pred_test, train_time, test_time


def check_and_remove_duplicates(distance_x, train_idx, verbose):
    """
    Checks for duplicate samples in the training set based on zero distances 
    and removes them if duplicates are found.

    Parameters:
    - distance_x: pd.DataFrame, structure (input) distance matrix.
    - train_idx: list, indices of training samples.

    Returns:
    - train_idx: list, updated list of training indices with duplicates removed.
    """
    # Check if any columns have more than one zero (excluding the diagonal)
    # First, we ensure the diagonal is ignored by temporarily setting it to a non-zero value (e.g., NaN)
    distance_matrix = distance_x.loc[train_idx, train_idx]
    dist_array = distance_matrix.values
    # check the dist_arrary
    zeros_count_per_column = np.sum(dist_array == 0, axis=0)
    # Check for columns with more than one zero
    if any(zeros_count_per_column > 1):
        if verbose:
            print("Duplicates detected, removing...")
        return remove_duplicate_samples(distance_x, train_idx, verbose)
    else:
        if verbose:
            print("No duplicates detected.")
        return train_idx
    
def remove_duplicate_samples(distance_x, train_idx, verbose):
    """
    Removes duplicate samples from the training set by identifying columns in the 
    distance matrix with zero distances to multiple rows.

    Parameters:
    - distance_x: pd.DataFrame, structure (input) distance matrix.
    - train_idx: list, indices of training samples.

    Returns:
    - filtered_train_idx: list, training indices after removing duplicates.
    """
    # Extract the submatrix of distance_x that corresponds to the training indices
    sub_distance_x = distance_x.loc[train_idx, train_idx]

    # Initialize an empty list to store indices of duplicate samples
    duplicate_indices = set()

    # Loop over each column in the submatrix
    for i, column in enumerate(sub_distance_x.columns):
        if column not in duplicate_indices:  # Skip columns already identified as duplicates
            # Get indices where the column has zero distance (excluding diagonal)
            zeros = sub_distance_x.index[sub_distance_x[column] == 0].tolist()
            # Exclude the diagonal element by removing 'column' from zeros
            if column in zeros:
                zeros.remove(column)
            # Add these indices to the set of duplicates
            duplicate_indices.update(zeros)

    # Create a set of all indices and remove the duplicates to find unique indices
    unique_indices = set(train_idx) - duplicate_indices
    #   
    num_removed = len(duplicate_indices)
    if verbose:
        print(f"Number of duplicate samples removed: {num_removed}")
    # print(duplicate_indices)
    # The result is already in the form of original indices from train_idx
    filtered_train_idx = list(unique_indices)

    return filtered_train_idx


def _rbf(x, s):
    return np.exp(-(x/s)**2)

def rbf(dist_array, response_train, anchors_idx, gamma=1):
    """
    Reconstruct the responses from the distances with a RBF function.

    Parameters:
    - dist_array: np.ndarray, distance array (n_anchors x n_test).
    - response_train: pd.Series, actual target values for training samples.
    - anchors_idx: list, indices of anchor points.
    - gamma: float, scale factor for the RBF kernel.

    Returns:
    - rt: np.ndarray, reconstructed responses for the test samples.
    """

    n_a, n_t = dist_array.shape
    response_real_values = response_train.loc[anchors_idx]
    if response_real_values.ndim == 1:
        response_real_values = response_real_values.values.reshape(-1, 1)
    if gamma is None:
        # gamma = 1 / response_real_values.shape[1]
        # gamma = np.mean(response_real_values.values.ravel(), axis=None)
        gamma = np.mean(dist_array, axis=None)

    rbf_v = np.vectorize(_rbf)
    k = rbf_v(dist_array, gamma).T  # rbf of distance. n_t x n_a
    # print(k.sum(axis=1))
    h = np.linalg.inv(np.diag(k.sum(axis=1)))  # normalize mat. n_test x n_test
    r = np.asarray(response_real_values)# .values  # real y. n_anchors x n_features.
    rt = h @ k @ r  # np.matmul. Does it work?
    
    return rt
    
def normalize_data(data, means=None, stds=None):
    """
    Normalizes data to zero mean and unit variance.

    Parameters:
    - data: np.ndarray, input data to normalize.
    - means: np.ndarray, pre-computed means (optional, defaults to the mean of `data`).
    - stds: np.ndarray, pre-computed standard deviations (optional, defaults to the std of `data`).

    Returns:
    - normalized_data: np.ndarray, normalized data.
    - means: np.ndarray, mean values used for normalization.
    - stds: np.ndarray, standard deviation values used for normalization.
    """
    # Calculate the mean and standard deviation of each column
    if means is None:
        means = np.mean(data, axis=0)
    if stds is None:
        stds = np.std(data, axis=0)
    
    # Avoid division by zero in case of zero standard deviation
    stds[stds == 0] = 1
    
    # Normalize the data
    normalized_data = (data - means) / stds
    
    return normalized_data, means, stds
