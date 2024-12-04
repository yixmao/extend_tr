import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.neighbors import NearestNeighbors



def select_anchor(distance_x, distance_y, train_idx, anchor_percentage, args):
    """
    Selects anchor points from the training samples.

    Parameters:
    - distance_x: pd.DataFrame, structure (input) distance matrix
    - distance_y: pd.DataFrame, response distance matrix
    - train_idx: list, indices of training samples
    - anchor_percentage: float, percentage of training samples to be used as anchors
    - args: Namespace, contains arguments

    Returns:
    - List of indices representing selected anchor points
    """

    # get the training samples
    distance_x_train = distance_x.loc[train_idx, train_idx]
    num_samples = len(distance_x_train)
    num_anchors = int(np.floor(num_samples * anchor_percentage))
    if num_anchors>args.max_num_anchors:
        num_anchors = args.max_num_anchors

    # select anchor points
    anchors_idx = apply_anchor_selection(distance_x_train, num_anchors, args)

    # refine the selected anchors with lasso - l1 regularization
    if args.refine_anchors_lasso:
        anchors_idx = refine_anchors_lasso(distance_x, distance_y, train_idx, anchors_idx, args.lasso_alpha, args.lasso_thres)

    return list(anchors_idx)

def apply_anchor_selection(distance_matrix, num_anchors, args):
    """
    Applies the specified anchor selection method to choose anchor points.

    Parameters:
    - distance_matrix: pd.DataFrame, structure (input) distance matrix between training samples
    - num_anchors: int, number of anchor points to select
    - args: Namespace, contains arguments

    Returns:
    - List of indices representing selected anchor points
    """
    if args.anchorselection == 'maximin':
        return select_anchor_maximin(distance_matrix, num_anchors)
    if args.anchorselection == 'maximin_density':
        return select_anchor_maximin_density(distance_matrix, num_anchors, args.weight_density)
    elif args.anchorselection == 'random':
        return distance_matrix.sample(n=num_anchors).index
    else:
        raise ValueError("Unsupported anchor selection method.")



def select_anchor_maximin(distance_matrix, num_anchors):
    """
    Selects anchor points using the Maximin strategy, maximizing the minimum distance 
    between anchor points.

    Parameters:
    - distance_matrix: pd.DataFrame, structure (input) distance matrix between training samples
    - num_anchors: int, number of anchor points to select

    Returns:
    - List of indices representing selected anchor points
    """

    # Ensure the distance matrix is a DataFrame
    if not isinstance(distance_matrix, pd.DataFrame):
        raise ValueError("distance_matrix must be a pandas DataFrame.")

    indices = distance_matrix.index  # Store the original indices of the DataFrame
    # Convert DataFrame to numpy array for distance calculations
    dist_array = distance_matrix.values

    initial_anchor_idx = np.random.choice(indices)
    anchors_idx = [initial_anchor_idx]  # List to store indices of selected anchors
    positions = [indices.get_loc(initial_anchor_idx)]  # List to store positions of selected anchors in dist_array

    # Iteratively select the rest of the anchors
    for _ in range(1, num_anchors):
        # Calculate the minimum distance from any chosen anchor to all other points
        min_distances = dist_array[positions, :].min(axis=0)  # Use positions for numpy operations
        
        # Find the index of the maximum value in min_distances
        max_min_dist_pos = min_distances.argmax()  # Position in the numpy array
        next_anchor_idx = indices[max_min_dist_pos]  # Corresponding DataFrame index

        # happens if all remaining samples have zero distance with a selected anchor point 
        if  min_distances[max_min_dist_pos] == 0:
            print(f"Stopping selection as the max-min distance is zero.")
            break
        else:
            # Update lists of positions and indices
            anchors_idx.append(next_anchor_idx)
            positions.append(max_min_dist_pos)
        # print(len(anchors_idx))
    return anchors_idx

def select_anchor_maximin_density(distance_matrix, num_anchors, weight_density, k_neighbors=20):
    """
    Selects anchor points using a combination of Maximin strategy and density-based weighting.

    Parameters:
    - distance_matrix: pd.DataFrame, structure (input) distance matrix between training samples
    - num_anchors: int, number of anchor points to select
    - weight_density: float, between 0 and 1: higher value represents more weight to the density
    - k_neighbors: int, number of neighbors used for density estimation

    Returns:
    - List of indices representing selected anchor points
    """

    # Ensure the distance matrix is a DataFrame
    if not isinstance(distance_matrix, pd.DataFrame):
        raise ValueError("distance_matrix must be a pandas DataFrame.")

    indices = distance_matrix.index  # Store the original indices of the DataFrame
    dist_array = distance_matrix.values  # Convert DataFrame to numpy array for distance calculations

    # Estimate the density using NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, metric='precomputed')
    nbrs.fit(dist_array)
    distances, _ = nbrs.kneighbors(dist_array)

    # Calculate density as the inverse of the mean distance to the k-nearest neighbors
    density = 1 / distances.mean(axis=1)  # Higher density corresponds to smaller mean distances

    # Normalize the density values by its min and max values
    density_normalized = (density - density.min()) / (density.max() - density.min())

    # Pick the first anchor with the highest density value
    initial_anchor_idx = density_normalized.argmax()  # Higher density values have higher priority
    anchors_idx = [indices[initial_anchor_idx]]  # List to store indices of selected anchors
    positions = [initial_anchor_idx]  # List to store positions of selected anchors in dist_array

    # Iteratively select the rest of the anchors
    for _ in range(1, num_anchors):
        # Calculate the minimum distance from any chosen anchor to all other points
        min_distances = dist_array[positions, :].min(axis=0)
        
        # Normalize the min_distance by its min and max values
        min_distances_normalized = (min_distances - min_distances.min()) / (min_distances.max() - min_distances.min())
        
        # Combine the normalized min_distance and density
        combined_metric = (1 - weight_density) * min_distances_normalized + weight_density * density_normalized

        # Sort the combined metric to find the best candidate that hasn't been picked
        sorted_indices = np.argsort(-combined_metric)  # Sort in descending order
        
        # Find the first unselected index
        for idx in sorted_indices:
            if indices[idx] not in anchors_idx:
                max_min_dist_pos = idx
                break

        next_anchor_idx = indices[max_min_dist_pos]

        # Stop if the min distance is zero (indicating identical points)
        if min_distances[max_min_dist_pos] == 0:
            print(f"Stopping selection as the max-min distance is zero.")
            break
        else:
            # Update lists of positions and indices
            anchors_idx.append(next_anchor_idx)
            positions.append(max_min_dist_pos)

    return anchors_idx

def refine_anchors_lasso(distance_x, distance_y, train_idx, anchors_idx, lasso_alpha, lasso_thres):
    """
    Refines selected anchor points using Lasso (L1 regularization) to only keep anchors
    with non-zero contributions.

    Parameters:
    - distance_x: pd.DataFrame, structure (input) distance matrix
    - distance_y: pd.DataFrame, response distance matrix
    - train_idx: list, indices of training samples
    - anchors_idx: list, initial anchor indices
    - lasso_alpha: controls the L1 penalty
    - lasso_thres: used to determine non-zero weights

    Returns:
    - List of indices representing refined anchor points
    """
    # sample the distances
    dist_x_train = distance_x.loc[train_idx, anchors_idx]
    dist_y_train = distance_y.loc[train_idx, anchors_idx]

    # lasso model
    lasso = Lasso(alpha=lasso_alpha)
    lasso.fit(dist_x_train, dist_y_train)

    # Analyze the weights
    weights = lasso.coef_
    # print(weights.shape)

    # Sum the absolute values over the columns
    column_sums = np.sum(np.abs(weights), axis=0)
    # print("Sum of absolute values of the weights over the columns:", column_sums)

    # 
    good_anchors = np.where(column_sums > lasso_thres)[0]

    # Calculate the percentage of good anchors - have weights not close to zeros
    percentage = len(good_anchors) / weights.shape[1]
    # print(f"Percentage of good anchors with nonzero weights: {percentage:.2f}")
    # extract the these anchors
    selected_anchors = [anchors_idx[i] for i in good_anchors]
    return selected_anchors

