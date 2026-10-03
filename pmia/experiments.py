"""Unified, reproducible experiment runner for all six original presets."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from .constants import IN, OUT
    from .datastructures import LabeledDataset
    from .metrics import compute_membership_metrics
    from .mia import MIAConfig, MIAEnvironment
    from .partial_mia import (
        DEFAULT_MAX_OUTPUT_BYTES,
        add_gaussian_noise,
        choose_feature_columns,
        enumerate_feature_values,
        feature_count_from_fraction,
        flip_noise_to_features,
        mask_features,
    )
else:
    from constants import IN, OUT
    from datastructures import LabeledDataset
    from metrics import compute_membership_metrics
    from mia import MIAConfig, MIAEnvironment
    from partial_mia import (
        DEFAULT_MAX_OUTPUT_BYTES,
        add_gaussian_noise,
        choose_feature_columns,
        enumerate_feature_values,
        feature_count_from_fraction,
        flip_noise_to_features,
        mask_features,
    )

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUPPORTED_MODALITIES = ("purchase", "cifar10", "newsgroups")
SUPPORTED_MODELS = ("lr", "mlp")
SUPPORTED_METHODS = (
    "baseline",
    "noise",
    "mask",
    "feature_selection",
    "enumerate",
)

PRESET_SIZES: dict[tuple[str, str], tuple[int, int]] = {
    ("purchase", "lr"): (2500, 499),
    ("purchase", "mlp"): (2500, 2500),
    ("cifar10", "lr"): (1000, 499),
    ("cifar10", "mlp"): (1000, 1000),
    ("newsgroups", "lr"): (1000, 499),
    ("newsgroups", "mlp"): (1000, 1000),
}


@dataclass(frozen=True, slots=True)
class ExperimentSettings:
    modality: str
    model_family: str
    training_size: int
    testing_size: int
    methods: tuple[str, ...] = (
        "baseline",
        "noise",
        "mask",
        "feature_selection",
    )
    target_fraction: float = 0.5
    training_proportion: float = 0.7
    num_shadow_models: int = 2
    seed: int = 0
    clusters: int = 2
    evaluation_size: int | None = None
    dataset_limit: int | None = None
    noise_level: float = 0.1
    feature_fraction: float = 0.05
    enumeration_fraction: float = 0.01
    mask_value: float | None = None
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES
    target_max_iter: int = 2000
    attack_max_iter: int = 2500
    hash_data: bool = True
    smoke: bool = False

    def __post_init__(self) -> None:
        if self.modality not in SUPPORTED_MODALITIES:
            raise ValueError(
                f"Unsupported modality {self.modality!r}; "
                f"choose from {SUPPORTED_MODALITIES}"
            )
        if self.model_family not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model family {self.model_family!r}; "
                f"choose from {SUPPORTED_MODELS}"
            )
        if not self.methods:
            raise ValueError("At least one experiment method is required")
        unknown_methods = sorted(set(self.methods) - set(SUPPORTED_METHODS))
        if unknown_methods:
            raise ValueError(f"Unsupported experiment methods: {unknown_methods}")
        if len(set(self.methods)) != len(self.methods):
            raise ValueError("Experiment methods must not contain duplicates")
        if "enumerate" in self.methods and self.modality != "purchase":
            raise ValueError(
                "Feature enumeration is only defined for binary Purchase inputs"
            )

        positive_integers = {
            "training_size": self.training_size,
            "testing_size": self.testing_size,
            "num_shadow_models": self.num_shadow_models,
            "clusters": self.clusters,
            "target_max_iter": self.target_max_iter,
            "attack_max_iter": self.attack_max_iter,
            "max_output_bytes": self.max_output_bytes,
        }
        for name, value in positive_integers.items():
            if not isinstance(value, (int, np.integer)) or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer; got {value}")
        if not isinstance(self.seed, (int, np.integer)):
            raise TypeError("seed must be an integer")
        if self.evaluation_size is not None and (
            not isinstance(self.evaluation_size, (int, np.integer))
            or self.evaluation_size < 1
        ):
            raise ValueError("evaluation_size must be a positive integer when provided")
        if self.dataset_limit is not None and (
            not isinstance(self.dataset_limit, (int, np.integer))
            or self.dataset_limit < 2
        ):
            raise ValueError("dataset_limit must be an integer of at least 2")
        numeric_fields = {
            "target_fraction": self.target_fraction,
            "training_proportion": self.training_proportion,
            "noise_level": self.noise_level,
            "feature_fraction": self.feature_fraction,
            "enumeration_fraction": self.enumeration_fraction,
        }
        for name, value in numeric_fields.items():
            if not isinstance(value, (int, float, np.integer, np.floating)):
                raise TypeError(f"{name} must be numeric")
        if not 0.0 < float(self.target_fraction) < 1.0:
            raise ValueError("target_fraction must be strictly between 0 and 1")
        if not 0.0 < float(self.training_proportion) < 1.0:
            raise ValueError("training_proportion must be strictly between 0 and 1")
        if not 0.0 <= float(self.noise_level) <= 1.0:
            raise ValueError("noise_level must be in [0, 1]")
        if not 0.0 < float(self.feature_fraction) <= 1.0:
            raise ValueError("feature_fraction must be in (0, 1]")
        if not 0.0 < float(self.enumeration_fraction) <= 1.0:
            raise ValueError("enumeration_fraction must be in (0, 1]")
        if self.mask_value is not None and (
            not isinstance(self.mask_value, (int, float, np.integer, np.floating))
            or not np.isfinite(self.mask_value)
        ):
            raise ValueError("mask_value must be finite when provided")
        if not isinstance(self.hash_data, (bool, np.bool_)):
            raise TypeError("hash_data must be boolean")
        if not isinstance(self.smoke, (bool, np.bool_)):
            raise TypeError("smoke must be boolean")
        if self.modality != "purchase" and self.clusters != 2:
            raise ValueError("--clusters only applies to the Purchase dataset")

        for name in positive_integers:
            object.__setattr__(self, name, int(getattr(self, name)))
        object.__setattr__(self, "seed", int(self.seed))
        if self.evaluation_size is not None:
            object.__setattr__(self, "evaluation_size", int(self.evaluation_size))
        if self.dataset_limit is not None:
            object.__setattr__(self, "dataset_limit", int(self.dataset_limit))
        for name in numeric_fields:
            object.__setattr__(self, name, float(getattr(self, name)))
        if self.mask_value is not None:
            object.__setattr__(self, "mask_value", float(self.mask_value))
        object.__setattr__(self, "hash_data", bool(self.hash_data))
        object.__setattr__(self, "smoke", bool(self.smoke))
        object.__setattr__(self, "methods", tuple(self.methods))


@dataclass(slots=True)
class DatasetBundle:
    dataset: LabeledDataset
    modality: str
    metadata: dict[str, Any]


def preset_settings(
    modality: str,
    model_family: str,
    *,
    smoke: bool = False,
    **overrides: Any,
) -> ExperimentSettings:
    if (modality, model_family) not in PRESET_SIZES:
        raise ValueError(
            f"No preset exists for modality={modality!r}, model={model_family!r}"
        )
    training_size, testing_size = PRESET_SIZES[(modality, model_family)]
    defaults: dict[str, Any] = {
        "modality": modality,
        "model_family": model_family,
        "training_size": 500 if smoke else training_size,
        "testing_size": 200 if smoke else testing_size,
        "dataset_limit": 5000 if smoke else None,
        "evaluation_size": 200 if smoke else None,
        "target_max_iter": 200 if smoke else 2000,
        "attack_max_iter": 300 if smoke else 2500,
        "smoke": smoke,
    }
    defaults.update({key: value for key, value in overrides.items() if value is not None})
    return ExperimentSettings(**defaults)


def _require_sklearn() -> None:
    try:
        installed_version = metadata.version("scikit-learn")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "scikit-learn is required for experiments but is not installed. "
            "Install requirements.txt in an isolated environment."
        ) from exc
    if not installed_version:
        raise RuntimeError("scikit-learn installation has no readable version")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise RuntimeError(f"Failed to hash data file {path}") from exc
    return digest.hexdigest()


def _update_array_hash(digest: Any, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(memoryview(contiguous).cast("B"))


def _dataset_fingerprint(dataset: LabeledDataset) -> str:
    digest = hashlib.sha256()
    if hasattr(dataset.X, "tocsr"):
        matrix = dataset.X.tocsr()
        digest.update(str(matrix.shape).encode("ascii"))
        _update_array_hash(digest, matrix.data)
        _update_array_hash(digest, matrix.indices)
        _update_array_hash(digest, matrix.indptr)
    else:
        _update_array_hash(digest, np.asarray(dataset.X))
    _update_array_hash(digest, np.asarray(dataset.y))
    return digest.hexdigest()


def _class_counts(y: np.ndarray) -> dict[str, int]:
    classes, counts = np.unique(y, return_counts=True)
    return {
        str(class_label): int(count)
        for class_label, count in zip(classes, counts)
    }


def _load_purchase(settings: ExperimentSettings) -> DatasetBundle:
    data_dir = PROJECT_ROOT / "datasets" / "purchase"
    feature_path = data_dir / "modified_purchase100.npy"
    label_path = data_dir / f"cluster_labels_{settings.clusters}.npy"
    for path in (feature_path, label_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required Purchase data file is missing: {path}")

    try:
        loaded = np.load(feature_path, allow_pickle=True)
        feature_object = loaded.item()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load trusted pickled Purchase features from {feature_path}"
        ) from exc
    if not isinstance(feature_object, dict) or set(feature_object) != {"features"}:
        raise ValueError(
            "Purchase feature artifact must be a dict containing only 'features'; "
            f"got keys {sorted(feature_object) if isinstance(feature_object, dict) else None}"
        )
    X = np.asarray(feature_object["features"])
    try:
        y = np.load(label_path, allow_pickle=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to load Purchase labels from {label_path}") from exc
    dataset = LabeledDataset(X, y)

    if dataset.X.dtype != np.int64:
        raise ValueError(
            f"Purchase features must be int64; got {dataset.X.dtype}"
        )
    if dataset.y.dtype != np.int32:
        raise ValueError(f"Purchase labels must be int32; got {dataset.y.dtype}")
    minimum = int(np.min(dataset.X))
    maximum = int(np.max(dataset.X))
    if minimum != 0 or maximum != 1:
        raise ValueError(
            "Purchase features must be binary with observed min=0 and max=1; "
            f"got min={minimum}, max={maximum}"
        )
    expected_labels = np.arange(settings.clusters)
    actual_labels = np.unique(dataset.y)
    if not np.array_equal(actual_labels, expected_labels):
        raise ValueError(
            f"Purchase labels are {actual_labels.tolist()}; expected "
            f"{expected_labels.tolist()}"
        )

    metadata: dict[str, Any] = {
        "source": "Purchase-100 local artifacts",
        "feature_path": str(feature_path),
        "label_path": str(label_path),
        "trusted_pickle_required": True,
        "source_shape": [int(value) for value in dataset.X.shape],
        "feature_dtype": str(dataset.X.dtype),
        "label_dtype": str(dataset.y.dtype),
    }
    if settings.hash_data:
        metadata["feature_file_sha256"] = _file_sha256(feature_path)
        metadata["label_file_sha256"] = _file_sha256(label_path)
    return DatasetBundle(dataset=dataset, modality="purchase", metadata=metadata)


def _load_cifar10(settings: ExperimentSettings) -> DatasetBundle:
    try:
        from tensorflow.keras.datasets import cifar10
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The CIFAR-10 experiment requires TensorFlow. Install "
            "requirements-image.txt in an isolated environment."
        ) from exc

    try:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    except Exception as exc:
        raise RuntimeError(
            "CIFAR-10 loading failed; verify the TensorFlow installation and "
            "network/cache availability"
        ) from exc
    X = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0).reshape(-1)
    X = X.reshape(X.shape[0], -1).astype(np.float32) / 255.0
    dataset = LabeledDataset(X, y)
    if dataset.X.shape != (60000, 3072):
        raise ValueError(f"Unexpected flattened CIFAR-10 shape {dataset.X.shape}")
    if not np.array_equal(np.unique(dataset.y), np.arange(10)):
        raise ValueError("CIFAR-10 labels are not exactly 0 through 9")
    return DatasetBundle(
        dataset=dataset,
        modality="cifar10",
        metadata={
            "source": "tensorflow.keras.datasets.cifar10",
            "source_shape": [60000, 32, 32, 3],
            "transformed_shape": [60000, 3072],
            "feature_dtype": str(dataset.X.dtype),
            "label_dtype": str(dataset.y.dtype),
            "normalization": "float32 divided by 255",
        },
    )


def _load_newsgroups(settings: ExperimentSettings) -> DatasetBundle:
    _require_sklearn()
    try:
        from sklearn.datasets import fetch_20newsgroups
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ModuleNotFoundError as exc:
        raise RuntimeError("scikit-learn text components are unavailable") from exc

    try:
        corpus = fetch_20newsgroups(
            subset="all",
            categories=None,
            remove=("headers", "footers", "quotes"),
        )
    except Exception as exc:
        raise RuntimeError(
            "20 Newsgroups loading failed; verify network/cache availability"
        ) from exc
    vectorizer = TfidfVectorizer(dtype=np.float64)
    try:
        X = vectorizer.fit_transform(corpus.data)
    except Exception as exc:
        raise RuntimeError("TF-IDF vectorization failed") from exc
    dataset = LabeledDataset(X.tocsr(), np.asarray(corpus.target))
    if not np.array_equal(np.unique(dataset.y), np.arange(20)):
        raise ValueError("20 Newsgroups labels are not exactly 0 through 19")
    return DatasetBundle(
        dataset=dataset,
        modality="newsgroups",
        metadata={
            "source": "sklearn.datasets.fetch_20newsgroups",
            "documents": len(corpus.data),
            "transformed_shape": [int(value) for value in dataset.X.shape],
            "sparse": True,
            "feature_dtype": str(dataset.X.dtype),
            "label_dtype": str(dataset.y.dtype),
            "removed_text": ["headers", "footers", "quotes"],
            "vocabulary_size": len(vectorizer.vocabulary_),
        },
    )


def load_dataset(settings: ExperimentSettings) -> DatasetBundle:
    loaders = {
        "purchase": _load_purchase,
        "cifar10": _load_cifar10,
        "newsgroups": _load_newsgroups,
    }
    bundle = loaders[settings.modality](settings)
    bundle.metadata["unlimited_class_counts"] = _class_counts(bundle.dataset.y)
    if settings.dataset_limit is not None:
        if settings.dataset_limit >= len(bundle.dataset):
            raise ValueError(
                f"dataset_limit={settings.dataset_limit} must be smaller than "
                f"the source dataset size {len(bundle.dataset)}"
            )
        bundle.dataset = bundle.dataset.sample(
            settings.dataset_limit,
            random_state=settings.seed,
            stratify=True,
        )
        bundle.metadata["reduction"] = {
            "status": "SMOKE/REDUCED",
            "limit": settings.dataset_limit,
            "sampling": "deterministic stratified sample without replacement",
        }

    bundle.metadata["experiment_shape"] = [
        int(value) for value in bundle.dataset.X.shape
    ]
    bundle.metadata["experiment_class_counts"] = _class_counts(bundle.dataset.y)
    if settings.hash_data:
        bundle.metadata["experiment_dataset_sha256"] = _dataset_fingerprint(
            bundle.dataset
        )
    return bundle


def _models(settings: ExperimentSettings) -> tuple[Any, Any, Any]:
    _require_sklearn()
    if settings.model_family == "lr":
        from sklearn.linear_model import LogisticRegression

        target = LogisticRegression(
            max_iter=settings.target_max_iter,
            random_state=settings.seed,
        )
        shadow = LogisticRegression(
            max_iter=settings.target_max_iter,
            random_state=settings.seed,
        )
        attack = LogisticRegression(
            max_iter=settings.attack_max_iter,
            random_state=settings.seed,
        )
        return target, shadow, attack

    from sklearn.neural_network import MLPClassifier

    target_hidden_layers = (
        (100, 50, 25) if settings.modality == "newsgroups" else (100,)
    )
    target = MLPClassifier(
        hidden_layer_sizes=target_hidden_layers,
        alpha=0.0001,
        max_iter=settings.target_max_iter,
        random_state=settings.seed,
    )
    shadow = MLPClassifier(
        hidden_layer_sizes=target_hidden_layers,
        alpha=0.0001,
        max_iter=settings.target_max_iter,
        random_state=settings.seed,
    )
    attack = MLPClassifier(
        hidden_layer_sizes=(100,),
        alpha=0.0001,
        max_iter=settings.attack_max_iter,
        random_state=settings.seed,
    )
    return target, shadow, attack


def _stack_reference_data(
    environment: MIAEnvironment,
) -> tuple[Any, np.ndarray]:
    matrices = [split.training_data.X for split in environment.shadow_data]
    labels = np.concatenate(
        [split.training_data.y for split in environment.shadow_data]
    )
    if any(hasattr(matrix, "tocsr") for matrix in matrices):
        if not all(hasattr(matrix, "tocsr") for matrix in matrices):
            raise RuntimeError("Cannot stack a mixture of sparse and dense matrices")
        try:
            from scipy import sparse
        except ModuleNotFoundError as exc:
            raise RuntimeError("Stacking sparse shadow data requires scipy") from exc
        return sparse.vstack(matrices, format="csr"), labels
    return np.concatenate([np.asarray(matrix) for matrix in matrices], axis=0), labels


def _least_informative_columns(
    bundle: DatasetBundle,
    environment: MIAEnvironment,
    count: int,
    *,
    seed: int,
) -> tuple[np.ndarray, str]:
    _require_sklearn()
    reference_X, reference_y = _stack_reference_data(environment)
    if bundle.modality == "newsgroups":
        from sklearn.feature_selection import chi2

        try:
            scores, _ = chi2(reference_X, reference_y)
        except Exception as exc:
            raise RuntimeError("Chi-squared text feature scoring failed") from exc
        selector = "chi2 on shadow-training data"
    else:
        from sklearn.feature_selection import mutual_info_classif

        try:
            scores = mutual_info_classif(
                reference_X,
                reference_y,
                discrete_features=bundle.modality == "purchase",
                random_state=seed,
            )
        except Exception as exc:
            raise RuntimeError("Mutual-information feature scoring failed") from exc
        selector = "mutual_info_classif on shadow-training data"

    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (environment.dataset.num_features,):
        raise RuntimeError(
            f"Feature scorer returned shape {scores.shape}; expected "
            f"({environment.dataset.num_features},)"
        )
    if not np.all(np.isfinite(scores)):
        bad_count = int(np.sum(~np.isfinite(scores)))
        raise RuntimeError(
            f"Feature scorer returned {bad_count} non-finite scores"
        )
    return np.sort(np.argsort(scores, kind="mergesort")[:count]), selector


def _selection_metadata(columns: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(columns.astype(np.int64, copy=False))
    return {
        "selected_feature_count": int(len(contiguous)),
        "selected_columns": [int(value) for value in contiguous],
        "selected_columns_sha256": hashlib.sha256(
            memoryview(contiguous).cast("B")
        ).hexdigest(),
    }


def _method_seed(base_seed: int, method: str) -> int:
    payload = f"partial-membership-nn:{int(base_seed)}:{method}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _evaluate_arrays(
    environment: MIAEnvironment,
    member_X: Any,
    non_member_X: Any,
) -> dict[str, Any]:
    member_decisions, member_probabilities = environment.is_member(member_X)
    non_member_decisions, non_member_probabilities = environment.is_member(
        non_member_X
    )
    y_true = np.concatenate(
        [
            np.full(len(member_decisions), IN, dtype=np.int64),
            np.full(len(non_member_decisions), OUT, dtype=np.int64),
        ]
    )
    decisions = np.concatenate([member_decisions, non_member_decisions])
    probabilities = np.concatenate(
        [member_probabilities, non_member_probabilities]
    )
    return compute_membership_metrics(
        y_true,
        decisions,
        probabilities,
    ).to_dict()


def _evaluate_enumeration(
    environment: MIAEnvironment,
    member_X: Any,
    non_member_X: Any,
    columns: np.ndarray,
    *,
    max_output_bytes: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    member_result = enumerate_feature_values(
        member_X,
        np.asarray([0, 1], dtype=np.int64),
        columns,
        max_output_bytes=max_output_bytes,
    )
    _, member_variant_probabilities = environment.is_member(member_result.X)
    member_probabilities = member_variant_probabilities.reshape(
        member_X.shape[0],
        member_result.variants_per_record,
    ).mean(axis=1)
    variants_per_record = member_result.variants_per_record
    del member_result, member_variant_probabilities

    non_member_result = enumerate_feature_values(
        non_member_X,
        np.asarray([0, 1], dtype=np.int64),
        columns,
        max_output_bytes=max_output_bytes,
    )
    if non_member_result.variants_per_record != variants_per_record:
        raise RuntimeError("Member/non-member enumeration sizes unexpectedly differ")
    _, non_member_variant_probabilities = environment.is_member(non_member_result.X)
    non_member_probabilities = non_member_variant_probabilities.reshape(
        non_member_X.shape[0],
        variants_per_record,
    ).mean(axis=1)

    member_decisions = member_probabilities >= 0.5
    non_member_decisions = non_member_probabilities >= 0.5
    y_true = np.concatenate(
        [
            np.full(len(member_decisions), IN, dtype=np.int64),
            np.full(len(non_member_decisions), OUT, dtype=np.int64),
        ]
    )
    metrics = compute_membership_metrics(
        y_true,
        np.concatenate([member_decisions, non_member_decisions]),
        np.concatenate([member_probabilities, non_member_probabilities]),
    ).to_dict()
    metadata = {
        "aggregation": "mean membership probability across every completion",
        "decision_threshold": 0.5,
        "variants_per_record": variants_per_record,
        "expanded_member_rows": int(member_X.shape[0] * variants_per_record),
        "expanded_non_member_rows": int(
            non_member_X.shape[0] * variants_per_record
        ),
    }
    return metrics, metadata


def _method_result(
    method: str,
    bundle: DatasetBundle,
    environment: MIAEnvironment,
    settings: ExperimentSettings,
    member_X: Any,
    non_member_X: Any,
    generator: np.random.Generator,
    method_seed: int,
) -> dict[str, Any]:
    if method == "baseline":
        return {
            "status": "PASS",
            "description": "Unmodified exact records",
            "metrics": _evaluate_arrays(environment, member_X, non_member_X),
            "metadata": {"method_seed": method_seed},
        }

    if method == "feature_selection":
        count = feature_count_from_fraction(
            environment.dataset.num_features,
            settings.feature_fraction,
        )
        columns, selector = _least_informative_columns(
            bundle,
            environment,
            count,
            seed=method_seed,
        )
    else:
        fraction = (
            settings.enumeration_fraction
            if method == "enumerate"
            else settings.feature_fraction
        )
        columns = choose_feature_columns(
            environment.dataset.num_features,
            fraction,
            random_state=generator,
        )
        selector = "deterministic random selection"

    metadata_for_method = _selection_metadata(columns)
    metadata_for_method["column_selection"] = selector
    metadata_for_method["method_seed"] = method_seed

    if method == "noise":
        if bundle.modality == "purchase":
            modified_member_X = flip_noise_to_features(
                member_X,
                columns,
                settings.noise_level,
                random_state=generator,
            )
            modified_non_member_X = flip_noise_to_features(
                non_member_X,
                columns,
                settings.noise_level,
                random_state=generator,
            )
            metadata_for_method["noise"] = {
                "type": "binary flip",
                "flip_probability": settings.noise_level,
            }
        else:
            modified_member_X = add_gaussian_noise(
                member_X,
                columns,
                settings.noise_level,
                random_state=generator,
                max_output_bytes=settings.max_output_bytes,
            )
            modified_non_member_X = add_gaussian_noise(
                non_member_X,
                columns,
                settings.noise_level,
                random_state=generator,
                max_output_bytes=settings.max_output_bytes,
            )
            metadata_for_method["noise"] = {
                "type": "Gaussian",
                "standard_deviation": settings.noise_level,
            }
        metrics = _evaluate_arrays(
            environment,
            modified_member_X,
            modified_non_member_X,
        )
    elif method in ("mask", "feature_selection"):
        mask_value = settings.mask_value
        if mask_value is None:
            mask_value = -1.0 if bundle.modality == "purchase" else 0.0
        modified_member_X = mask_features(
            member_X,
            columns,
            mask_value=mask_value,
        )
        modified_non_member_X = mask_features(
            non_member_X,
            columns,
            mask_value=mask_value,
        )
        metadata_for_method["mask_value"] = mask_value
        metrics = _evaluate_arrays(
            environment,
            modified_member_X,
            modified_non_member_X,
        )
    elif method == "enumerate":
        metrics, enumeration_metadata = _evaluate_enumeration(
            environment,
            member_X,
            non_member_X,
            columns,
            max_output_bytes=settings.max_output_bytes,
        )
        metadata_for_method.update(enumeration_metadata)
    else:
        raise RuntimeError(f"Method dispatch is missing for {method!r}")

    return {
        "status": "PASS",
        "description": {
            "noise": "Selected features perturbed with explicit random noise",
            "mask": "Same randomly selected feature mask applied to both groups",
            "feature_selection": (
                "Least-informative shadow-derived features masked identically "
                "for both groups"
            ),
            "enumerate": (
                "Every binary completion aggregated back to one result per record"
            ),
        }[method],
        "metrics": metrics,
        "metadata": metadata_for_method,
    }


def _runtime_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    for package in ("scipy", "scikit-learn", "pandas", "tensorflow"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def run_experiment(settings: ExperimentSettings) -> dict[str, Any]:
    _require_sklearn()
    bundle = load_dataset(settings)
    target_model, shadow_model, attack_model = _models(settings)
    config = MIAConfig.from_proportions(
        len(bundle.dataset),
        settings.target_fraction,
        settings.training_size,
        settings.testing_size,
        settings.training_proportion,
    )
    environment = MIAEnvironment(
        bundle.dataset,
        target_model,
        shadow_model,
        settings.num_shadow_models,
        attack_model,
        config,
        random_state=settings.seed,
    )

    try:
        from sklearn.exceptions import ConvergenceWarning
    except ModuleNotFoundError as exc:
        raise RuntimeError("scikit-learn warning definitions are unavailable") from exc

    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        warnings.simplefilter("error", RuntimeWarning)
        environment.train_basic_models()
        target_score, shadow_scores = environment.test_basic_models()
        environment.train_attack_models()
        attack_scores = environment.test_attack_models()

    maximum_evaluation_size = min(
        len(environment.target_data.training_data),
        len(environment.target_data.testing_data),
    )
    evaluation_size = (
        maximum_evaluation_size
        if settings.evaluation_size is None
        else settings.evaluation_size
    )
    if evaluation_size > maximum_evaluation_size:
        raise ValueError(
            f"evaluation_size={evaluation_size} exceeds the balanced maximum "
            f"{maximum_evaluation_size}"
        )
    member_X = environment.target_data.training_data.X[:evaluation_size]
    non_member_X = environment.target_data.testing_data.X[:evaluation_size]
    method_results: dict[str, Any] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for method in settings.methods:
            method_seed = _method_seed(settings.seed, method)
            method_results[method] = _method_result(
                method,
                bundle,
                environment,
                settings,
                member_X,
                non_member_X,
                np.random.default_rng(method_seed),
                method_seed,
            )

    mode = "SMOKE/REDUCED" if settings.dataset_limit is not None else "FULL_DATASET"
    return {
        "schema_version": 1,
        "status": "PASS",
        "mode": mode,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "settings": asdict(settings),
        "runtime": _runtime_versions(),
        "dataset": bundle.metadata,
        "splits": {
            "target_training_records": len(environment.target_data.training_data),
            "target_testing_records": len(environment.target_data.testing_data),
            "shadow_models": len(environment.shadow_data),
            "shadow_training_records_each": [
                len(split.training_data) for split in environment.shadow_data
            ],
            "shadow_testing_records_each": [
                len(split.testing_data) for split in environment.shadow_data
            ],
            "evaluation_members": evaluation_size,
            "evaluation_non_members": evaluation_size,
        },
        "model_scores": {
            "target_test_accuracy": target_score,
            "shadow_test_accuracy": shadow_scores,
            "attack_test_accuracy_by_target_class": attack_scores,
        },
        "methods": method_results,
        "scientific_scope": {
            "target_models": 1,
            "seed_count": 1,
            "warning": (
                "One run is not a paper-level estimate. Repeat independent seeds "
                "and aggregate uncertainty before drawing conclusions."
            ),
        },
    }


def save_result(result: dict[str, Any], output_path: Path) -> None:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, output_path)
    except Exception as exc:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise RuntimeError(f"Failed to atomically save results to {output_path}") from exc


def print_result(result: dict[str, Any]) -> None:
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
