"""Corrected shadow-model membership inference environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

if __package__:
    from .constants import IN, OUT
    from .datastructures import (
        ClassifierList,
        LabeledDataset,
        RandomState,
        TrainTestSplit,
        clone_classifier,
        validate_classifier,
    )
else:
    from constants import IN, OUT
    from datastructures import (
        ClassifierList,
        LabeledDataset,
        RandomState,
        TrainTestSplit,
        clone_classifier,
        validate_classifier,
    )


@dataclass(frozen=True, slots=True)
class LabelMapping:
    """Minimal explicit replacement for the mutating LabelEncoder usage."""

    classes_: NDArray[Any]

    def inverse_transform(self, labels: ArrayLike) -> NDArray[Any]:
        encoded = np.asarray(labels)
        if encoded.ndim != 1:
            raise ValueError("Encoded labels must be one-dimensional")
        if not np.issubdtype(encoded.dtype, np.integer):
            raise TypeError("Encoded labels must be integers")
        if np.any(encoded < 0) or np.any(encoded >= len(self.classes_)):
            raise ValueError("Encoded label is outside the known class range")
        return self.classes_[encoded]


@dataclass(frozen=True, slots=True)
class MIAConfig:
    absolute_target_size: int
    absolute_training_size: int
    absolute_testing_size: int
    training_proportion: float = 0.7

    def __post_init__(self) -> None:
        integer_fields = (
            "absolute_target_size",
            "absolute_training_size",
            "absolute_testing_size",
        )
        for field_name in integer_fields:
            value = getattr(self, field_name)
            if not isinstance(value, (int, np.integer)) or int(value) < 1:
                raise ValueError(f"{field_name} must be a positive integer; got {value}")
        if not isinstance(self.training_proportion, (int, float, np.floating)):
            raise TypeError("training_proportion must be numeric")
        if not 0.0 < float(self.training_proportion) < 1.0:
            raise ValueError("training_proportion must be strictly between 0 and 1")

    def validate_num_samples(self, num_samples: int) -> None:
        if not isinstance(num_samples, (int, np.integer)) or int(num_samples) < 2:
            raise ValueError("num_samples must be an integer of at least 2")
        if not 0 < self.absolute_target_size < int(num_samples):
            raise ValueError(
                "absolute_target_size must leave at least one record for both "
                f"target and shadow partitions; got target={self.absolute_target_size}, "
                f"total={num_samples}"
            )

    def validate_pools(
        self,
        target_pool: TrainTestSplit,
        shadow_pool: TrainTestSplit,
        *,
        num_classes: int,
    ) -> None:
        if self.absolute_training_size < num_classes:
            raise ValueError(
                "absolute_training_size must be at least the number of classes "
                f"({num_classes}) for class-specific attacks"
            )
        if self.absolute_testing_size < num_classes:
            raise ValueError(
                "absolute_testing_size must be at least the number of classes "
                f"({num_classes}) for class-specific attacks"
            )

        pools = {
            "target training": (
                self.absolute_training_size,
                len(target_pool.training_data),
            ),
            "target testing": (
                self.absolute_testing_size,
                len(target_pool.testing_data),
            ),
            "shadow training": (
                self.absolute_training_size,
                len(shadow_pool.training_data),
            ),
            "shadow testing": (
                self.absolute_testing_size,
                len(shadow_pool.testing_data),
            ),
        }
        for name, (requested, available) in pools.items():
            if requested > available:
                raise ValueError(
                    f"Requested {requested} records from the {name} pool, but only "
                    f"{available} are available"
                )

    @staticmethod
    def from_proportions(
        num_samples: int,
        relative_target_size: float,
        absolute_training_size: int,
        absolute_testing_size: int,
        training_proportion: float = 0.7,
    ) -> "MIAConfig":
        if not isinstance(relative_target_size, (int, float, np.floating)):
            raise TypeError("relative_target_size must be numeric")
        if not 0.0 < float(relative_target_size) < 1.0:
            raise ValueError("relative_target_size must be strictly between 0 and 1")
        absolute_target_size = int(np.floor(num_samples * float(relative_target_size)))
        config = MIAConfig(
            absolute_target_size=absolute_target_size,
            absolute_training_size=absolute_training_size,
            absolute_testing_size=absolute_testing_size,
            training_proportion=training_proportion,
        )
        config.validate_num_samples(num_samples)
        return config


def _encode_labels(dataset: LabeledDataset) -> tuple[LabeledDataset, LabelMapping]:
    try:
        classes, encoded = np.unique(dataset.y, return_inverse=True)
    except Exception as exc:
        raise ValueError("Labels could not be deterministically encoded") from exc
    if classes.size < 2:
        raise ValueError("Membership inference requires at least two target classes")
    encoded_dataset = LabeledDataset(
        dataset.X,
        encoded.astype(np.int64),
        record_ids=dataset.record_ids.copy(),
    )
    return encoded_dataset, LabelMapping(classes_=classes)


def _normalize_query_features(X: Any, expected_features: int) -> Any:
    features = X if hasattr(X, "shape") else np.asarray(X)
    shape = getattr(features, "shape", None)
    if shape is None:
        raise ValueError("Membership query features do not expose a shape")

    if len(shape) == 1:
        features = np.asarray(features).reshape(1, -1)
        shape = features.shape
    if len(shape) != 2:
        raise ValueError(f"Membership query must be 1-D or 2-D; got shape {shape}")
    if shape[0] < 1:
        raise ValueError("Membership query must contain at least one record")
    if shape[1] != expected_features:
        raise ValueError(
            f"Membership query has {shape[1]} features; expected {expected_features}"
        )
    return features


def _canonical_probabilities(
    model: Any,
    X: Any,
    expected_labels: NDArray[np.int64],
    *,
    context: str,
) -> NDArray[np.float64]:
    try:
        raw = model.predict_proba(X)
    except Exception as exc:
        raise RuntimeError(f"{context}: predict_proba failed") from exc

    probabilities = np.asarray(raw, dtype=np.float64)
    expected_rows = int(X.shape[0])
    if probabilities.ndim != 2:
        raise RuntimeError(
            f"{context}: predict_proba must return a 2-D matrix; "
            f"got {probabilities.shape}"
        )
    if probabilities.shape[0] != expected_rows:
        raise RuntimeError(
            f"{context}: predict_proba returned {probabilities.shape[0]} rows "
            f"for {expected_rows} inputs"
        )
    if not np.all(np.isfinite(probabilities)):
        raise RuntimeError(f"{context}: predict_proba returned NaN or infinity")
    if np.any(probabilities < -1e-12) or np.any(probabilities > 1.0 + 1e-12):
        raise RuntimeError(f"{context}: probability values are outside [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6):
        raise RuntimeError(f"{context}: probability rows do not sum to one")

    classes = getattr(model, "classes_", None)
    if classes is None:
        raise RuntimeError(f"{context}: fitted model does not expose classes_")
    classes_array = np.asarray(classes)
    if classes_array.ndim != 1 or len(classes_array) != probabilities.shape[1]:
        raise RuntimeError(
            f"{context}: classes_ shape {classes_array.shape} does not match "
            f"{probabilities.shape[1]} probability columns"
        )

    column_order: list[int] = []
    for label in expected_labels:
        matches = np.flatnonzero(classes_array == label)
        if matches.size != 1:
            raise RuntimeError(
                f"{context}: expected class {int(label)} exactly once in classes_="
                f"{classes_array.tolist()}"
            )
        column_order.append(int(matches[0]))

    if len(column_order) != probabilities.shape[1]:
        raise RuntimeError(
            f"{context}: model classes {classes_array.tolist()} do not exactly match "
            f"expected classes {expected_labels.tolist()}"
        )
    return probabilities[:, column_order]


class MIAEnvironment:
    """Train and evaluate a class-specific shadow-model membership attack."""

    def __init__(
        self,
        dataset: LabeledDataset,
        target_model: Any,
        shadow_model_base: Any,
        num_shadow_models: int,
        attack_model_base: Any,
        config: MIAConfig,
        random_state: RandomState = 0,
    ) -> None:
        if not isinstance(dataset, LabeledDataset):
            raise TypeError("dataset must be a LabeledDataset")
        if not isinstance(config, MIAConfig):
            raise TypeError("config must be an MIAConfig")
        if not isinstance(num_shadow_models, (int, np.integer)):
            raise TypeError("num_shadow_models must be an integer")
        if int(num_shadow_models) < 1:
            raise ValueError("num_shadow_models must be at least 1")
        validate_classifier(target_model, name="target_model")
        validate_classifier(shadow_model_base, name="shadow_model_base")
        validate_classifier(attack_model_base, name="attack_model_base")

        if not isinstance(random_state, (int, np.integer, np.random.Generator)):
            raise TypeError(
                "random_state must be an integer or numpy.random.Generator"
            )
        self.random_state = random_state
        self._generator = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(int(random_state))
        )

        encoded_dataset, label_mapping = _encode_labels(dataset)
        self.dataset = encoded_dataset
        self.label_encoder = label_mapping
        self.num_classes = len(label_mapping.classes_)
        self._target_labels = np.arange(self.num_classes, dtype=np.int64)
        self._membership_labels = np.asarray([OUT, IN], dtype=np.int64)

        config.validate_num_samples(len(encoded_dataset))
        target_partition, shadow_partition = encoded_dataset.split_size(
            config.absolute_target_size,
            random_state=self._generator,
            stratify=True,
        )
        target_pool = target_partition.split(
            config.training_proportion,
            random_state=self._generator,
        )
        shadow_pool = shadow_partition.split(
            config.training_proportion,
            random_state=self._generator,
        )
        config.validate_pools(
            target_pool,
            shadow_pool,
            num_classes=self.num_classes,
        )

        self.target_data = target_pool.sample(
            config.absolute_training_size,
            config.absolute_testing_size,
            random_state=self._generator,
            stratify=True,
        )
        self.shadow_data = shadow_pool.sample_n(
            config.absolute_training_size,
            config.absolute_testing_size,
            int(num_shadow_models),
            random_state=self._generator,
            stratify=True,
        )
        self._validate_sampled_class_coverage()

        self.target_model = clone_classifier(target_model, name="target_model")
        self.shadow_models = ClassifierList(
            clone_classifier(
                shadow_model_base,
                name=f"shadow_model_base[{index}]",
            )
            for index in range(int(num_shadow_models))
        )
        self.attack_models = ClassifierList(
            clone_classifier(
                attack_model_base,
                name=f"attack_model_base[class={class_index}]",
            )
            for class_index in range(self.num_classes)
        )

        self.attack_data: list[TrainTestSplit] | None = None
        self._basic_training_attempted = False
        self._attack_training_attempted = False
        self.trained_basic_models = False
        self.trained_attack_models = False
        self.config = config

    def _validate_sampled_class_coverage(self) -> None:
        expected = set(range(self.num_classes))
        datasets: list[tuple[str, LabeledDataset]] = [
            ("target training", self.target_data.training_data),
            ("target testing", self.target_data.testing_data),
        ]
        for model_index, split in enumerate(self.shadow_data):
            datasets.append((f"shadow {model_index} training", split.training_data))
            datasets.append((f"shadow {model_index} testing", split.testing_data))

        for name, dataset in datasets:
            actual = set(int(value) for value in np.unique(dataset.y))
            if actual != expected:
                missing = sorted(expected - actual)
                unexpected = sorted(actual - expected)
                raise ValueError(
                    f"{name} does not contain the exact encoded class set; "
                    f"missing={missing}, unexpected={unexpected}"
                )

    def train_basic_models(self) -> None:
        if self._basic_training_attempted:
            raise RuntimeError(
                "Basic-model training has already been attempted; construct a new "
                "environment instead of reusing possibly partially fitted models"
            )
        self._basic_training_attempted = True
        try:
            self.target_model.fit(
                self.target_data.training_data.X,
                self.target_data.training_data.y,
            )
        except Exception as exc:
            raise RuntimeError("Training the target model failed") from exc

        self.shadow_models.train(
            [split.training_data for split in self.shadow_data]
        )
        self._validate_fitted_target_classes()
        for index, model in enumerate(self.shadow_models.models):
            self._validate_fitted_target_classes(
                model=model,
                context=f"shadow model {index}",
            )
        self.trained_basic_models = True

    def _validate_fitted_target_classes(
        self,
        *,
        model: Any | None = None,
        context: str = "target model",
    ) -> None:
        fitted_model = self.target_model if model is None else model
        classes = getattr(fitted_model, "classes_", None)
        if classes is None:
            raise RuntimeError(f"{context} does not expose classes_ after fitting")
        classes_array = np.asarray(classes)
        if (
            classes_array.ndim != 1
            or len(classes_array) != len(self._target_labels)
            or not np.array_equal(np.sort(classes_array), self._target_labels)
        ):
            raise RuntimeError(
                f"{context} classes are {classes_array.tolist()}; expected "
                f"exactly the set {self._target_labels.tolist()}"
            )

    def test_basic_models(self) -> tuple[float, list[float]]:
        if not self.trained_basic_models:
            raise RuntimeError("Basic models must be trained before testing")
        try:
            target_score = float(
                self.target_model.score(
                    self.target_data.testing_data.X,
                    self.target_data.testing_data.y,
                )
            )
        except Exception as exc:
            raise RuntimeError("Testing the target model failed") from exc
        if not np.isfinite(target_score):
            raise RuntimeError(f"Target model returned non-finite score {target_score}")
        shadow_scores = self.shadow_models.test(
            [split.testing_data for split in self.shadow_data]
        )
        return target_score, shadow_scores

    def _build_attack_datasets(
        self,
        models: Sequence[Any],
        datasets: Sequence[TrainTestSplit],
        *,
        data_size: int,
        source_name: str,
    ) -> list[LabeledDataset]:
        if len(models) != len(datasets):
            raise ValueError(
                f"{source_name}: model and dataset counts differ "
                f"({len(models)} != {len(datasets)})"
            )

        in_features: list[list[NDArray[np.float64]]] = [
            [] for _ in range(self.num_classes)
        ]
        out_features: list[list[NDArray[np.float64]]] = [
            [] for _ in range(self.num_classes)
        ]

        for model_index, (model, split) in enumerate(zip(models, datasets)):
            sampled = split.sample(
                data_size,
                data_size,
                random_state=self._generator,
                stratify=True,
            )
            in_probabilities = _canonical_probabilities(
                model,
                sampled.training_data.X,
                self._target_labels,
                context=f"{source_name} model {model_index} IN records",
            )
            out_probabilities = _canonical_probabilities(
                model,
                sampled.testing_data.X,
                self._target_labels,
                context=f"{source_name} model {model_index} OUT records",
            )
            in_routing_classes = np.argmax(in_probabilities, axis=1)
            out_routing_classes = np.argmax(out_probabilities, axis=1)

            for class_index in range(self.num_classes):
                # Route training examples with the same information available to
                # is_member(): the model's predicted class, not the hidden true label.
                in_mask = in_routing_classes == class_index
                out_mask = out_routing_classes == class_index
                if np.any(in_mask):
                    in_features[class_index].append(in_probabilities[in_mask])
                if np.any(out_mask):
                    out_features[class_index].append(out_probabilities[out_mask])

        attack_datasets: list[LabeledDataset] = []
        for class_index in range(self.num_classes):
            if not in_features[class_index]:
                raise RuntimeError(
                    f"{source_name}: class {class_index} has no IN posterior records"
                )
            if not out_features[class_index]:
                raise RuntimeError(
                    f"{source_name}: class {class_index} has no OUT posterior records"
                )

            class_in = np.concatenate(in_features[class_index], axis=0)
            class_out = np.concatenate(out_features[class_index], axis=0)
            balanced_size = min(len(class_in), len(class_out))
            if balanced_size < 1:
                raise RuntimeError(
                    f"{source_name}: class {class_index} cannot form a balanced "
                    "membership dataset"
                )
            in_indices = self._generator.choice(
                len(class_in),
                size=balanced_size,
                replace=False,
            )
            out_indices = self._generator.choice(
                len(class_out),
                size=balanced_size,
                replace=False,
            )
            X = np.concatenate(
                [class_in[in_indices], class_out[out_indices]],
                axis=0,
            )
            y = np.concatenate(
                [
                    np.full(balanced_size, IN, dtype=np.int64),
                    np.full(balanced_size, OUT, dtype=np.int64),
                ]
            )
            permutation = self._generator.permutation(len(y))
            attack_datasets.append(LabeledDataset(X[permutation], y[permutation]))

        return attack_datasets

    def train_attack_models(self) -> None:
        if not self.trained_basic_models:
            raise RuntimeError("Basic models must be trained before attack models")
        if self._attack_training_attempted:
            raise RuntimeError(
                "Attack-model training has already been attempted; construct a new "
                "environment instead of reusing possibly partially fitted models"
            )
        self._attack_training_attempted = True

        data_size = min(
            self.config.absolute_training_size,
            self.config.absolute_testing_size,
        )
        attack_training = self._build_attack_datasets(
            self.shadow_models.models,
            self.shadow_data,
            data_size=data_size,
            source_name="shadow attack training",
        )
        attack_testing = self._build_attack_datasets(
            [self.target_model],
            [self.target_data],
            data_size=data_size,
            source_name="target attack evaluation",
        )
        self.attack_data = [
            TrainTestSplit(training, testing)
            for training, testing in zip(attack_training, attack_testing)
        ]
        self.attack_models.train(
            [split.training_data for split in self.attack_data]
        )
        for class_index, model in enumerate(self.attack_models.models):
            classes = getattr(model, "classes_", None)
            classes_array = None if classes is None else np.asarray(classes)
            if (
                classes_array is None
                or classes_array.ndim != 1
                or len(classes_array) != len(self._membership_labels)
                or not np.array_equal(
                    np.sort(classes_array),
                    self._membership_labels,
                )
            ):
                raise RuntimeError(
                    f"Attack model {class_index} classes are "
                    f"{None if classes_array is None else classes_array.tolist()}; "
                    f"expected exactly the set {self._membership_labels.tolist()}"
                )
        self.trained_attack_models = True

    def test_attack_models(self) -> list[float]:
        if not self.trained_attack_models or self.attack_data is None:
            raise RuntimeError("Attack models must be trained before testing")
        return self.attack_models.test(
            [split.testing_data for split in self.attack_data]
        )

    def is_member(
        self,
        X: Any,
    ) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
        if not self.trained_attack_models:
            raise RuntimeError("Attack models must be trained before membership queries")

        features = _normalize_query_features(
            X,
            expected_features=self.dataset.num_features,
        )
        target_probabilities = _canonical_probabilities(
            self.target_model,
            features,
            self._target_labels,
            context="target membership query",
        )
        predicted_classes = np.argmax(target_probabilities, axis=1).astype(np.int64)

        decisions = np.empty(features.shape[0], dtype=np.bool_)
        membership_probabilities = np.empty(features.shape[0], dtype=np.float64)

        for class_index in np.unique(predicted_classes):
            row_indices = np.flatnonzero(predicted_classes == class_index)
            class_attack_probabilities = _canonical_probabilities(
                self.attack_models[int(class_index)],
                target_probabilities[row_indices],
                self._membership_labels,
                context=f"attack membership query for class {int(class_index)}",
            )
            predicted_membership_labels = np.argmax(
                class_attack_probabilities,
                axis=1,
            )
            decisions[row_indices] = predicted_membership_labels == IN
            membership_probabilities[row_indices] = class_attack_probabilities[:, IN]

        if not np.all(np.isfinite(membership_probabilities)):
            raise RuntimeError("Membership query produced non-finite probabilities")
        return decisions, membership_probabilities
