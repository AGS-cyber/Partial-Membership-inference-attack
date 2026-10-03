import numpy as np
from numpy.typing import ArrayLike
import random
import string
import itertools



def product_dataset(
    dataset: ArrayLike,
    range: ArrayLike,
    column_number: int,
    return_original_column=False,
):
    """
    This helper method receives a dataset (a 2d array)
    and will produce a form of combinatorial product of the dataset and the range (a 1d array).
    It will return a 2d array that has `len(range) * len(dataset)` rows.

    Specifically, it will produce a dataset where each original row is repeated `len(range)` times,
    and the column at `column_number` takes every possible value from the range.
    """
    # Create a dataset where each row is repeated |range| times.
    repeated_rows = np.repeat(dataset, len(range), axis=0)
    # Create a new column that repeats the full range vector as many times
    # as there were rows in the original data.
    new_column = np.tile(range, len(dataset))
    # Replace column.
    original_column = np.copy(repeated_rows[:, column_number])
    repeated_rows[:, column_number] = new_column

    if return_original_column:
        return repeated_rows, original_column

    return repeated_rows

def modified_product_dataset(
    dataset: ArrayLike,
    range: ArrayLike,
    column_percentage: float,
    return_original_columns=False,
):
    """
    This helper method receives a dataset (a 2d array)
    and will produce a form of combinatorial product of the dataset and the range (a 1d array).
    It will return a 2d array that has `len(range) ** num_columns * len(dataset)` rows.

    Specifically, it will produce a dataset where each original row is repeated `len(range) ** num_columns` times,
    and the columns based on the `column_percentage` take every possible value from the range.
    """
    num_columns = int(dataset.shape[1] * column_percentage)
    column_indices = np.random.choice(dataset.shape[1], num_columns, replace=False)

    # Create new columns using itertools.product
    new_columns = np.array(list(itertools.product(range, repeat=num_columns)))

    # Create a dataset where each row is repeated |range| ** num_columns times.
    repeated_rows = np.repeat(dataset, new_columns.shape[0], axis=0)

    # Replace columns based on column_indices.
    original_columns = np.copy(repeated_rows[:, column_indices])
    repeated_rows[:, column_indices] = np.tile(new_columns, (len(dataset), 1))

    if return_original_columns:
        return repeated_rows, original_columns

    return repeated_rows

def closest_match_distance(
    X: ArrayLike, datapoint: ArrayLike, distance_columns: ArrayLike
) -> float:
    # Separate distance columns
    target_columns = X[:, distance_columns]
    target_values = datapoint[distance_columns]

    remaining_X = np.delete(X, distance_columns, axis=1)
    remaining_datapoint = np.delete(datapoint, distance_columns)

    # Find matches
    matching_rows = (remaining_X == remaining_datapoint).all(axis=1)

    target_matching = target_columns[matching_rows]
    try:
        return np.min(np.sum(np.abs(target_matching - target_values), axis=1))
    except ValueError:
        return None
    
def permute_features(dataset, column_numbers):
    """
    Randomly shuffles the values of specified features (columns) across the dataset.
    
    Parameters:
    - dataset: np.ndarray, the dataset to be modified.
    - column_numbers: list of int, the indices of the columns (features) to be permuted.
    
    Returns:
    - permuted_dataset: np.ndarray, the dataset with the specified features permuted.
    """
    permuted_dataset = np.copy(dataset)
    for column_number in column_numbers:
        np.random.shuffle(permuted_dataset[:, column_number])
    return permuted_dataset

def flip_noise_to_features(dataset, column_numbers, flip_prob=0.1):
    """
    Randomly flips values in specified features (columns) of a binary dataset to simulate noise.
    
    Parameters:
    - dataset: np.ndarray, the dataset to be modified. Assumes binary features (0s and 1s).
    - column_numbers: list of int, the indices of the columns (features) where values will be flipped.
    - flip_prob: float, the probability of flipping each value in the specified columns.
    
    Returns:
    - noisy_dataset: np.ndarray, the dataset with values flipped in the specified features.
    """
    # Copy dataset to avoid modifying the original
    noisy_dataset = np.copy(dataset)
    # Iterate over specified columns to add noise
    for column_number in column_numbers:
        # For each column, determine which values to flip based on flip_prob
        flip_mask = np.random.rand(noisy_dataset.shape[0]) < flip_prob
        # Flip the values: 0s become 1s and 1s become 0s
        noisy_dataset[flip_mask, column_number] = 1 - noisy_dataset[flip_mask, column_number]
    return noisy_dataset

def add_gaussian_noise(dataset, column_numbers, noise_level=0.1):
    noisy_dataset = np.copy(dataset).astype(np.float64)
    for column_number in column_numbers:
        noise = np.random.normal(loc=0.0, scale=noise_level, size=(dataset.shape[0],))
        noisy_dataset[:, column_number] += noise
    return noisy_dataset


