"""Validated dataset and classifier containers for membership inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

RandomState = int | np.random.Generator


def _rng(random_state: RandomState) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    if not isinstance(random_state, (int, np.integer)):
        raise TypeError(
            "random_state must be an integer or numpy.random.Generator; "
            f"got {type(random_state).__name__}"
        )
    return np.random.default_rng(int(random_state))


def _normalize_labels(y: ArrayLike) -> NDArray[Any]:
    labels = np.asarray(y)
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.ndim != 1:
        raise ValueError(
            "Labels must have shape (n_samples,) or (n_samples, 1); "
            f"got {labels.shape}"
        )
    if labels.size == 0:
        raise ValueError("Labels must not be empty")
    return labels


def _normalize_features(X: Any) -> Any:
    features = X if hasattr(X, "shape") else np.asarray(X)
    shape = getattr(features, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError(
            "Features must be a two-dimensional matrix with shape "
            f"(n_samples, n_features); got {shape}"
        )
    if shape[0] == 0:
        raise ValueError("Features must contain at least one sample")
    if shape[1] == 0:
        raise ValueError("Features must contain at least one feature")
    return features


def _allocate_stratified_counts(
    counts: NDArray[np.int64],
    sample_size: int,
    *,
    require_remainder_classes: bool,
) -> NDArray[np.int64]:
    num_classes = len(counts)
    total = int(counts.sum())
    minimum = np.ones(num_classes, dtype=np.int64)
    maximum = counts - 1 if require_remainder_classes else counts.copy()

    if np.any(maximum < minimum):
        raise ValueError(
            "Every class must contain at least two records for this stratified split"
            if require_remainder_classes
            else "Every class must contain at least one record"
        )
    if sample_size < int(minimum.sum()) or sample_size > int(maximum.sum()):
        remainder_requirement = (
            " while preserving every class in the remainder"
            if require_remainder_classes
            else ""
        )
        raise ValueError(
            f"Cannot draw {sample_size} stratified records from {total} records "
            f"across {num_classes} classes{remainder_requirement}; valid range is "
            f"[{int(minimum.sum())}, {int(maximum.sum())}]"
        )

    ideal = counts.astype(np.float64) * (sample_size / total)
    allocation = np.floor(ideal).astype(np.int64)
    allocation = np.maximum(allocation, minimum)
    allocation = np.minimum(allocation, maximum)

    while int(allocation.sum()) < sample_size:
        candidates = np.flatnonzero(allocation < maximum)
        if candidates.size == 0:
            raise RuntimeError("Stratified allocation could not reach the sample size")
        priorities = ideal[candidates] - allocation[candidates]
        chosen = int(candidates[np.argmax(priorities)])
        allocation[chosen] += 1

    while int(allocation.sum()) > sample_size:
        candidates = np.flatnonzero(allocation > minimum)
        if candidates.size == 0:
            raise RuntimeError("Stratified allocation could not reduce to the sample size")
        priorities = allocation[candidates] - ideal[candidates]
        chosen = int(candidates[np.argmax(priorities)])
        allocation[chosen] -= 1

    return allocation


def _stratified_indices(
    y: NDArray[Any],
    sample_size: int,
    random_state: RandomState,
    *,
    require_remainder_classes: bool,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    generator = _rng(random_state)
    classes, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
    if len(classes) < 2:
        raise ValueError("Stratified sampling requires at least two classes")

    allocation = _allocate_stratified_counts(
        counts.astype(np.int64),
        sample_size,
        require_remainder_classes=require_remainder_classes,
    )
    selected_parts: list[NDArray[np.int64]] = []
    remainder_parts: list[NDArray[np.int64]] = []

    for class_index, take_count in enumerate(allocation):
        indices = np.flatnonzero(inverse == class_index).astype(np.int64)
        generator.shuffle(indices)
        selected_parts.append(indices[: int(take_count)])
        remainder_parts.append(indices[int(take_count) :])

    selected = np.concatenate(selected_parts)
    remainder = np.concatenate(remainder_parts)
    generator.shuffle(selected)
    generator.shuffle(remainder)
    return selected, remainder


@dataclass(slots=True)
class LabeledDataset:
    """A validated feature matrix and its one-dimensional labels."""

    X: Any
    y: NDArray[Any]
    record_ids: NDArray[np.int64] | None = None

    def __post_init__(self) -> None:
        self.X = _normalize_features(self.X)
        self.y = _normalize_labels(self.y)
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(
                "Features and labels must contain the same number of samples; "
                f"got {self.X.shape[0]} and {self.y.shape[0]}"
            )

        if self.record_ids is None:
            self.record_ids = np.arange(self.y.shape[0], dtype=np.int64)
        else:
            record_ids = np.asarray(self.record_ids)
            if record_ids.ndim != 1 or record_ids.shape[0] != self.y.shape[0]:
                raise ValueError(
                    "record_ids must be one-dimensional and match the dataset length"
                )
            self.record_ids = record_ids.astype(np.int64, copy=False)

    @staticmethod
    def from_pandas(df: Any, label_column: str) -> "LabeledDataset":
        if label_column not in df.columns:
            raise KeyError(f"Label column {label_column!r} is not present")
        return LabeledDataset(
            df.drop(label_column, axis=1).to_numpy(),
            df[label_column].to_numpy(),
        )

    def __len__(self) -> int:
        return int(self.y.shape[0])

    @property
    def num_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def num_classes(self) -> int:
        return int(np.unique(self.y).size)

    def take(self, indices: Sequence[int] | NDArray[np.integer[Any]]) -> "LabeledDataset":
        index_array = np.asarray(indices, dtype=np.int64)
        if index_array.ndim != 1:
            raise ValueError("Dataset indices must be one-dimensional")
        if index_array.size == 0:
            raise ValueError("Cannot create an empty dataset selection")
        return LabeledDataset(
            self.X[index_array],
            self.y[index_array],
            self.record_ids[index_array],
        )

    def split_at(self, index: int) -> tuple["LabeledDataset", "LabeledDataset"]:
        if not isinstance(index, (int, np.integer)):
            raise TypeError("Split index must be an integer")
        if not 0 < int(index) < len(self):
            raise ValueError(f"Split index must be in [1, {len(self) - 1}]; got {index}")
        first_indices = np.arange(int(index), dtype=np.int64)
        second_indices = np.arange(int(index), len(self), dtype=np.int64)
        return self.take(first_indices), self.take(second_indices)

    def split_size(
        self,
        first_size: int,
        *,
        random_state: RandomState = 0,
        stratify: bool = True,
    ) -> tuple["LabeledDataset", "LabeledDataset"]:
        if not isinstance(first_size, (int, np.integer)):
            raise TypeError("first_size must be an integer")
        first_size = int(first_size)
        if not 0 < first_size < len(self):
            raise ValueError(
                f"first_size must be in [1, {len(self) - 1}]; got {first_size}"
            )

        generator = _rng(random_state)
        if stratify:
            first_indices, second_indices = _stratified_indices(
                self.y,
                first_size,
                generator,
                require_remainder_classes=True,
            )
        else:
            permutation = generator.permutation(len(self))
            first_indices = permutation[:first_size]
            second_indices = permutation[first_size:]
        return self.take(first_indices), self.take(second_indices)

    def split(
        self,
        training_proportion: float = 0.8,
        *,
        random_state: RandomState = 0,
    ) -> "TrainTestSplit":
        if not isinstance(training_proportion, (float, int, np.floating)):
            raise TypeError("training_proportion must be numeric")
        training_proportion = float(training_proportion)
        if not 0.0 < training_proportion < 1.0:
            raise ValueError("training_proportion must be strictly between 0 and 1")
        training_size = int(np.floor(len(self) * training_proportion))
        training, testing = self.split_size(
            training_size,
            random_state=random_state,
            stratify=True,
        )
        return TrainTestSplit(training, testing)

    def sample(
        self,
        n_samples: int,
        *,
        random_state: RandomState = 0,
        stratify: bool = False,
    ) -> "LabeledDataset":
        if not isinstance(n_samples, (int, np.integer)):
            raise TypeError("n_samples must be an integer")
        n_samples = int(n_samples)
        if not 0 < n_samples <= len(self):
            raise ValueError(
                f"n_samples must be in [1, {len(self)}]; got {n_samples}"
            )

        generator = _rng(random_state)
        if n_samples == len(self):
            indices = generator.permutation(len(self))
        elif stratify:
            indices, _ = _stratified_indices(
                self.y,
                n_samples,
                generator,
                require_remainder_classes=False,
            )
        else:
            indices = generator.choice(len(self), size=n_samples, replace=False)
        return self.take(indices)


@dataclass(slots=True)
class TrainTestSplit:
    training_data: LabeledDataset
    testing_data: LabeledDataset

    def __post_init__(self) -> None:
        if self.training_data.num_features != self.testing_data.num_features:
            raise ValueError(
                "Training and testing data must have the same feature count; "
                f"got {self.training_data.num_features} and "
                f"{self.testing_data.num_features}"
            )

    def sample(
        self,
        training_size: int,
        testing_size: int,
        *,
        random_state: RandomState = 0,
        stratify: bool = True,
    ) -> "TrainTestSplit":
        generator = _rng(random_state)
        return TrainTestSplit(
            self.training_data.sample(
                training_size,
                random_state=generator,
                stratify=stratify,
            ),
            self.testing_data.sample(
                testing_size,
                random_state=generator,
                stratify=stratify,
            ),
        )

    def sample_n(
        self,
        training_size: int,
        testing_size: int,
        num_sets: int,
        *,
        random_state: RandomState = 0,
        stratify: bool = True,
    ) -> list["TrainTestSplit"]:
        if not isinstance(num_sets, (int, np.integer)) or int(num_sets) < 1:
            raise ValueError("num_sets must be a positive integer")

        generator = _rng(random_state)
        sampled: list[TrainTestSplit] = []
        seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
        for _ in range(int(num_sets)):
            split = self.sample(
                training_size,
                testing_size,
                random_state=generator,
                stratify=stratify,
            )
            signature = (
                tuple(sorted(int(value) for value in split.training_data.record_ids)),
                tuple(sorted(int(value) for value in split.testing_data.record_ids)),
            )
            if signature in seen:
                raise RuntimeError(
                    "Independent shadow sampling produced an exactly duplicated "
                    "train/test record set"
                )
            seen.add(signature)
            sampled.append(split)
        return sampled


def validate_classifier(model: Any, *, name: str) -> None:
    missing = [
        method
        for method in ("fit", "predict_proba", "score")
        if not callable(getattr(model, method, None))
    ]
    if missing:
        raise TypeError(
            f"{name} must implement fit, predict_proba, and score; missing {missing}"
        )


def clone_classifier(model: Any, *, name: str) -> Any:
    custom_clone = getattr(model, "clone_for_mia", None)
    if callable(custom_clone):
        cloned = custom_clone()
        validate_classifier(cloned, name=f"clone of {name}")
        return cloned

    try:
        from sklearn.base import clone
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Cannot clone {name}: scikit-learn is not installed and the model "
            "does not implement clone_for_mia()"
        ) from exc

    try:
        cloned = clone(model)
    except Exception as exc:
        raise RuntimeError(f"Failed to clone {name}") from exc
    validate_classifier(cloned, name=f"clone of {name}")
    return cloned


class ClassifierList:
    def __init__(self, models: Iterable[Any]) -> None:
        self.models = list(models)
        if not self.models:
            raise ValueError("ClassifierList requires at least one model")
        for index, model in enumerate(self.models):
            validate_classifier(model, name=f"models[{index}]")

    def train(self, datasets: Sequence[LabeledDataset]) -> None:
        if len(datasets) != len(self.models):
            raise ValueError(
                "Model and dataset counts must match; "
                f"got {len(self.models)} models and {len(datasets)} datasets"
            )
        for index, (model, dataset) in enumerate(zip(self.models, datasets)):
            try:
                model.fit(dataset.X, dataset.y)
            except Exception as exc:
                raise RuntimeError(f"Training classifier {index} failed") from exc

    def test(self, datasets: Sequence[LabeledDataset]) -> list[float]:
        if len(datasets) != len(self.models):
            raise ValueError(
                "Model and dataset counts must match; "
                f"got {len(self.models)} models and {len(datasets)} datasets"
            )
        results: list[float] = []
        for index, (model, dataset) in enumerate(zip(self.models, datasets)):
            try:
                score = float(model.score(dataset.X, dataset.y))
            except Exception as exc:
                raise RuntimeError(f"Testing classifier {index} failed") from exc
            if not np.isfinite(score):
                raise RuntimeError(f"Classifier {index} returned non-finite score {score}")
            results.append(score)
        return results

    def __len__(self) -> int:
        return len(self.models)

    def __getitem__(self, key: int) -> Any:
        return self.models[key]
