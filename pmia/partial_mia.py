"""Deterministic, validated partial-feature transformations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

if __package__:
    from .datastructures import RandomState
else:
    from datastructures import RandomState

DEFAULT_MAX_OUTPUT_BYTES = 512 * 1024 * 1024


def _rng(random_state: RandomState) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    if not isinstance(random_state, (int, np.integer)):
        raise TypeError(
            "random_state must be an integer or numpy.random.Generator; "
            f"got {type(random_state).__name__}"
        )
    return np.random.default_rng(int(random_state))


def _matrix_shape(dataset: Any) -> tuple[int, int]:
    shape = getattr(dataset, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError(f"dataset must be a 2-D matrix; got shape {shape}")
    if shape[0] < 1 or shape[1] < 1:
        raise ValueError(f"dataset must be nonempty; got shape {shape}")
    return int(shape[0]), int(shape[1])


def _dense_matrix(dataset: ArrayLike, *, operation: str) -> NDArray[Any]:
    if hasattr(dataset, "tocsr"):
        raise TypeError(
            f"{operation} requires a dense matrix; refusing to silently densify "
            "a sparse input"
        )
    matrix = np.asarray(dataset)
    _matrix_shape(matrix)
    return matrix


def _columns(
    column_numbers: Sequence[int] | NDArray[np.integer[Any]],
    num_features: int,
) -> NDArray[np.int64]:
    columns = np.asarray(column_numbers, dtype=np.int64)
    if columns.ndim != 1:
        raise ValueError("column_numbers must be one-dimensional")
    if columns.size == 0:
        raise ValueError("At least one feature column must be selected")
    if np.any(columns < 0) or np.any(columns >= num_features):
        raise IndexError(
            f"Feature columns must be in [0, {num_features - 1}]; "
            f"got {columns.tolist()}"
        )
    if np.unique(columns).size != columns.size:
        raise ValueError("Feature columns must not contain duplicates")
    return columns


def feature_count_from_fraction(num_features: int, fraction: float) -> int:
    if not isinstance(num_features, (int, np.integer)) or int(num_features) < 1:
        raise ValueError("num_features must be a positive integer")
    if not isinstance(fraction, (int, float, np.floating)):
        raise TypeError("fraction must be numeric")
    fraction = float(fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in the interval (0, 1]")
    return int(np.ceil(int(num_features) * fraction))


def choose_feature_columns(
    num_features: int,
    fraction: float,
    *,
    random_state: RandomState = 0,
) -> NDArray[np.int64]:
    count = feature_count_from_fraction(num_features, fraction)
    return np.sort(
        _rng(random_state).choice(num_features, size=count, replace=False)
    ).astype(np.int64)


@dataclass(frozen=True, slots=True)
class EnumerationResult:
    X: NDArray[Any]
    selected_columns: NDArray[np.int64]
    original_values: NDArray[Any]
    variants_per_record: int


def estimate_enumeration(
    dataset: ArrayLike,
    value_range: ArrayLike,
    selected_column_count: int,
) -> tuple[int, int]:
    matrix = _dense_matrix(dataset, operation="Feature enumeration")
    values = np.asarray(value_range)
    if values.ndim != 1 or values.size < 1:
        raise ValueError("value_range must be a nonempty one-dimensional array")
    if not isinstance(selected_column_count, (int, np.integer)):
        raise TypeError("selected_column_count must be an integer")
    if not 1 <= int(selected_column_count) <= matrix.shape[1]:
        raise ValueError(
            "selected_column_count must be between 1 and the number of features"
        )

    variants = int(values.size) ** int(selected_column_count)
    output_rows = int(matrix.shape[0]) * variants
    output_bytes = output_rows * int(matrix.shape[1]) * int(matrix.dtype.itemsize)
    return output_rows, output_bytes


def enumerate_feature_values(
    dataset: ArrayLike,
    value_range: ArrayLike,
    column_numbers: Sequence[int] | NDArray[np.integer[Any]],
    *,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> EnumerationResult:
    matrix = _dense_matrix(dataset, operation="Feature enumeration")
    columns = _columns(column_numbers, matrix.shape[1])
    values = np.asarray(value_range)
    if values.ndim != 1 or values.size < 1:
        raise ValueError("value_range must be a nonempty one-dimensional array")
    if not isinstance(max_output_bytes, (int, np.integer)) or max_output_bytes < 1:
        raise ValueError("max_output_bytes must be a positive integer")

    _, estimated_bytes = estimate_enumeration(matrix, values, len(columns))
    if estimated_bytes > int(max_output_bytes):
        raise MemoryError(
            "Feature enumeration refused: estimated dense output is "
            f"{estimated_bytes:,} bytes, above the configured "
            f"{int(max_output_bytes):,}-byte limit"
        )

    grids = np.meshgrid(*([values] * len(columns)), indexing="ij")
    combinations = np.stack([grid.reshape(-1) for grid in grids], axis=1)
    variants = int(combinations.shape[0])
    repeated = np.repeat(matrix, variants, axis=0)
    repeated[:, columns] = np.tile(combinations, (len(matrix), 1))
    return EnumerationResult(
        X=repeated,
        selected_columns=columns.copy(),
        original_values=matrix[:, columns].copy(),
        variants_per_record=variants,
    )


def product_dataset(
    dataset: ArrayLike,
    value_range: ArrayLike,
    column_number: int,
    return_original_column: bool = False,
    *,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> NDArray[Any] | tuple[NDArray[Any], NDArray[Any]]:
    result = enumerate_feature_values(
        dataset,
        value_range,
        [column_number],
        max_output_bytes=max_output_bytes,
    )
    if return_original_column:
        repeated_original = np.repeat(
            result.original_values[:, 0],
            result.variants_per_record,
        )
        return result.X, repeated_original
    return result.X


def modified_product_dataset(
    dataset: ArrayLike,
    value_range: ArrayLike,
    column_percentage: float,
    return_original_columns: bool = False,
    *,
    random_state: RandomState = 0,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> NDArray[Any] | tuple[NDArray[Any], NDArray[Any]]:
    matrix = _dense_matrix(dataset, operation="Feature enumeration")
    columns = choose_feature_columns(
        matrix.shape[1],
        column_percentage,
        random_state=random_state,
    )
    result = enumerate_feature_values(
        matrix,
        value_range,
        columns,
        max_output_bytes=max_output_bytes,
    )
    if return_original_columns:
        repeated_original = np.repeat(
            result.original_values,
            result.variants_per_record,
            axis=0,
        )
        return result.X, repeated_original
    return result.X


def closest_match_distance(
    X: ArrayLike,
    datapoint: ArrayLike,
    distance_columns: Sequence[int] | NDArray[np.integer[Any]],
) -> float:
    matrix = _dense_matrix(X, operation="Closest-match distance")
    point = np.asarray(datapoint)
    if point.ndim != 1 or point.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"datapoint must have shape ({matrix.shape[1]},); got {point.shape}"
        )
    columns = _columns(distance_columns, matrix.shape[1])
    remaining_columns = np.setdiff1d(
        np.arange(matrix.shape[1], dtype=np.int64),
        columns,
        assume_unique=True,
    )
    matching_rows = np.all(
        matrix[:, remaining_columns] == point[remaining_columns],
        axis=1,
    )
    if not np.any(matching_rows):
        raise LookupError(
            "No row matches the datapoint on all non-distance columns"
        )
    distances = np.sum(
        np.abs(matrix[matching_rows][:, columns] - point[columns]),
        axis=1,
    )
    minimum = float(np.min(distances))
    if not np.isfinite(minimum):
        raise RuntimeError("Closest-match distance is not finite")
    return minimum


def permute_features(
    dataset: ArrayLike,
    column_numbers: Sequence[int] | NDArray[np.integer[Any]],
    *,
    random_state: RandomState = 0,
) -> NDArray[Any]:
    matrix = _dense_matrix(dataset, operation="Feature permutation")
    columns = _columns(column_numbers, matrix.shape[1])
    generator = _rng(random_state)
    permuted = matrix.copy()
    for column in columns:
        permuted[:, column] = permuted[generator.permutation(len(permuted)), column]
    return permuted


def flip_noise_to_features(
    dataset: ArrayLike,
    column_numbers: Sequence[int] | NDArray[np.integer[Any]],
    flip_prob: float = 0.1,
    *,
    random_state: RandomState = 0,
) -> NDArray[Any]:
    matrix = _dense_matrix(dataset, operation="Binary flip noise")
    columns = _columns(column_numbers, matrix.shape[1])
    if not isinstance(flip_prob, (int, float, np.floating)):
        raise TypeError("flip_prob must be numeric")
    flip_prob = float(flip_prob)
    if not 0.0 <= flip_prob <= 1.0:
        raise ValueError("flip_prob must be in the interval [0, 1]")
    selected_values = matrix[:, columns]
    if not np.all((selected_values == 0) | (selected_values == 1)):
        raise ValueError(
            "Binary flip noise requires every selected value to be exactly 0 or 1"
        )

    generator = _rng(random_state)
    noisy = matrix.copy()
    mask = generator.random((matrix.shape[0], len(columns))) < flip_prob
    noisy[:, columns] = np.where(mask, 1 - selected_values, selected_values)
    return noisy


def add_gaussian_noise(
    dataset: ArrayLike,
    column_numbers: Sequence[int] | NDArray[np.integer[Any]],
    noise_level: float = 0.1,
    *,
    random_state: RandomState = 0,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> Any:
    rows, num_features = _matrix_shape(dataset)
    columns = _columns(column_numbers, num_features)
    if not isinstance(noise_level, (int, float, np.floating)):
        raise TypeError("noise_level must be numeric")
    noise_level = float(noise_level)
    if noise_level < 0.0:
        raise ValueError("noise_level must be nonnegative")
    if not isinstance(max_output_bytes, (int, np.integer)) or max_output_bytes < 1:
        raise ValueError("max_output_bytes must be a positive integer")

    generator = _rng(random_state)
    if hasattr(dataset, "tocsr"):
        try:
            from scipy import sparse
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Sparse Gaussian noise requires scipy, which is not installed"
            ) from exc

        inserted_values = rows * len(columns)
        estimated_bytes = inserted_values * 24
        if estimated_bytes > int(max_output_bytes):
            raise MemoryError(
                "Sparse Gaussian noise refused: selected columns would introduce "
                f"approximately {estimated_bytes:,} bytes, above the configured "
                f"{int(max_output_bytes):,}-byte limit"
            )
        row_indices = np.repeat(np.arange(rows, dtype=np.int64), len(columns))
        column_indices = np.tile(columns, rows)
        values = generator.normal(
            loc=0.0,
            scale=noise_level,
            size=inserted_values,
        )
        noise = sparse.coo_matrix(
            (values, (row_indices, column_indices)),
            shape=(rows, num_features),
        ).tocsr()
        result = dataset.astype(np.float64).tocsr() + noise
        if not np.all(np.isfinite(result.data)):
            raise RuntimeError("Gaussian noise produced NaN or infinity")
        return result

    matrix = np.asarray(dataset)
    noisy = matrix.astype(np.float64, copy=True)
    noisy[:, columns] += generator.normal(
        loc=0.0,
        scale=noise_level,
        size=(rows, len(columns)),
    )
    if not np.all(np.isfinite(noisy)):
        raise RuntimeError("Gaussian noise produced NaN or infinity")
    return noisy


def mask_features(
    dataset: ArrayLike,
    column_numbers: Sequence[int] | NDArray[np.integer[Any]],
    *,
    mask_value: float,
) -> Any:
    _, num_features = _matrix_shape(dataset)
    columns = _columns(column_numbers, num_features)
    if not np.isfinite(mask_value):
        raise ValueError("mask_value must be finite")

    if hasattr(dataset, "tocsr"):
        if mask_value != 0:
            raise ValueError(
                "Sparse feature masking only supports mask_value=0; a nonzero "
                "sentinel would densify the selected columns"
            )
        column_mask = np.ones(num_features, dtype=np.float64)
        column_mask[columns] = 0.0
        masked = dataset.multiply(column_mask).tocsr()
        masked.eliminate_zeros()
        return masked

    matrix = np.asarray(dataset)
    masked = matrix.copy()
    masked[:, columns] = mask_value
    return masked
