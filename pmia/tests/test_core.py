from __future__ import annotations

import json
import unittest
from dataclasses import asdict

import numpy as np

from pmia.datastructures import LabeledDataset
from pmia.experiments import ExperimentSettings, preset_settings
from pmia.metrics import UndefinedMetricError, compute_membership_metrics
from pmia.mia import MIAConfig, MIAEnvironment
from pmia.partial_mia import (
    add_gaussian_noise,
    closest_match_distance,
    enumerate_feature_values,
    flip_noise_to_features,
    mask_features,
    modified_product_dataset,
)


class NearestCentroidClassifier:
    """Small deterministic probability classifier used only by unit tests."""

    def __init__(self, *, reverse_classes: bool = False) -> None:
        self.reverse_classes = reverse_classes

    def clone_for_mia(self) -> "NearestCentroidClassifier":
        return NearestCentroidClassifier(reverse_classes=self.reverse_classes)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NearestCentroidClassifier":
        matrix = np.asarray(X, dtype=np.float64)
        labels = np.asarray(y)
        classes = np.unique(labels)
        if classes.size < 2:
            raise ValueError("Test classifier requires at least two classes")
        if self.reverse_classes:
            classes = classes[::-1]
        self.classes_ = classes
        self.centroids_ = np.stack(
            [matrix[labels == class_label].mean(axis=0) for class_label in classes]
        )
        if not np.all(np.isfinite(self.centroids_)):
            raise ValueError("Test classifier produced non-finite centroids")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        matrix = np.asarray(X, dtype=np.float64)
        squared_distances = np.sum(
            (matrix[:, None, :] - self.centroids_[None, :, :]) ** 2,
            axis=2,
        )
        logits = -squared_distances
        logits -= logits.max(axis=1, keepdims=True)
        exponentials = np.exp(logits)
        return exponentials / exponentials.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))


def synthetic_dataset(seed: int = 12) -> LabeledDataset:
    generator = np.random.default_rng(seed)
    class_zero = generator.normal(
        loc=(-2.0, -0.5),
        scale=(0.7, 0.8),
        size=(240, 2),
    )
    class_one = generator.normal(
        loc=(2.0, 0.5),
        scale=(0.7, 0.8),
        size=(240, 2),
    )
    X = np.concatenate([class_zero, class_one], axis=0)
    y = np.asarray(["left"] * len(class_zero) + ["right"] * len(class_one))
    permutation = generator.permutation(len(y))
    return LabeledDataset(X[permutation], y[permutation])


def environment(*, reverse_classes: bool = False) -> MIAEnvironment:
    model = NearestCentroidClassifier(reverse_classes=reverse_classes)
    return MIAEnvironment(
        synthetic_dataset(),
        model,
        model,
        3,
        model,
        MIAConfig.from_proportions(
            480,
            relative_target_size=0.5,
            absolute_training_size=100,
            absolute_testing_size=50,
        ),
        random_state=77,
    )


class DatasetTests(unittest.TestCase):
    def test_rejects_invalid_shapes_and_lengths(self) -> None:
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            LabeledDataset(np.ones(3), np.asarray([0, 1, 0]))
        with self.assertRaisesRegex(ValueError, "same number"):
            LabeledDataset(np.ones((3, 2)), np.asarray([0, 1]))
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            LabeledDataset(np.ones((1, 2)), np.asarray([]))

    def test_shadow_sampling_is_distinct_and_reproducible(self) -> None:
        dataset = synthetic_dataset()
        pool = dataset.split(0.7, random_state=5)
        first = pool.sample_n(80, 40, 3, random_state=9, stratify=True)
        second = pool.sample_n(80, 40, 3, random_state=9, stratify=True)

        first_signatures = [
            (
                tuple(sorted(split.training_data.record_ids)),
                tuple(sorted(split.testing_data.record_ids)),
            )
            for split in first
        ]
        second_signatures = [
            (
                tuple(sorted(split.training_data.record_ids)),
                tuple(sorted(split.testing_data.record_ids)),
            )
            for split in second
        ]
        self.assertEqual(first_signatures, second_signatures)
        self.assertEqual(len(set(first_signatures)), 3)


class EnvironmentTests(unittest.TestCase):
    def test_does_not_mutate_labels_and_uses_disjoint_partitions(self) -> None:
        dataset = synthetic_dataset()
        original_labels = dataset.y.copy()
        model = NearestCentroidClassifier()
        env = MIAEnvironment(
            dataset,
            model,
            model,
            2,
            model,
            MIAConfig.from_proportions(480, 0.5, 100, 50),
            random_state=4,
        )
        np.testing.assert_array_equal(dataset.y, original_labels)

        target_ids = set(env.target_data.training_data.record_ids) | set(
            env.target_data.testing_data.record_ids
        )
        for split in env.shadow_data:
            shadow_ids = set(split.training_data.record_ids) | set(
                split.testing_data.record_ids
            )
            self.assertTrue(target_ids.isdisjoint(shadow_ids))

    def test_lifecycle_errors_are_explicit(self) -> None:
        env = environment()
        with self.assertRaisesRegex(RuntimeError, "Attack models must be trained"):
            env.is_member(np.asarray([0.0, 0.0]))
        with self.assertRaisesRegex(RuntimeError, "Basic models must be trained"):
            env.train_attack_models()
        with self.assertRaisesRegex(RuntimeError, "Basic models must be trained"):
            env.test_basic_models()

    def test_queries_preserve_order_and_support_single_class_batches(self) -> None:
        env = environment(reverse_classes=True)
        env.train_basic_models()
        target_score, shadow_scores = env.test_basic_models()
        self.assertTrue(np.isfinite(target_score))
        self.assertEqual(len(shadow_scores), 3)
        env.train_attack_models()
        self.assertEqual(len(env.test_attack_models()), 2)

        rows = np.asarray(
            [
                [-2.5, -0.2],
                [2.3, 0.1],
                [-1.8, -1.0],
                [1.7, 1.1],
                [2.0, -0.2],
                [-2.1, 0.4],
            ]
        )
        batch_decisions, batch_probabilities = env.is_member(rows)
        single_results = [env.is_member(row) for row in rows]
        single_decisions = np.asarray(
            [bool(result[0][0]) for result in single_results],
            dtype=np.bool_,
        )
        single_probabilities = np.asarray(
            [float(result[1][0]) for result in single_results],
            dtype=np.float64,
        )

        self.assertEqual(batch_decisions.dtype, np.bool_)
        self.assertEqual(batch_probabilities.dtype, np.float64)
        np.testing.assert_array_equal(batch_decisions, single_decisions)
        np.testing.assert_allclose(
            batch_probabilities,
            single_probabilities,
            rtol=0,
            atol=1e-12,
        )

    def test_rejects_wrong_feature_count(self) -> None:
        env = environment()
        env.train_basic_models()
        env.train_attack_models()
        with self.assertRaisesRegex(ValueError, "expected 2"):
            env.is_member(np.ones((2, 3)))

    def test_training_attempts_cannot_reuse_fitted_state(self) -> None:
        env = environment()
        env.train_basic_models()
        with self.assertRaisesRegex(RuntimeError, "already been attempted"):
            env.train_basic_models()
        env.train_attack_models()
        with self.assertRaisesRegex(RuntimeError, "already been attempted"):
            env.train_attack_models()


class PartialFeatureTests(unittest.TestCase):
    def test_noise_is_deterministic_and_non_mutating(self) -> None:
        X = np.asarray([[0, 1, 0], [1, 0, 1], [0, 0, 1]], dtype=np.int64)
        original = X.copy()
        first = flip_noise_to_features(
            X,
            [0, 2],
            flip_prob=0.5,
            random_state=3,
        )
        second = flip_noise_to_features(
            X,
            [0, 2],
            flip_prob=0.5,
            random_state=3,
        )
        np.testing.assert_array_equal(X, original)
        np.testing.assert_array_equal(first, second)

        gaussian = add_gaussian_noise(
            X,
            [1],
            noise_level=0.2,
            random_state=2,
        )
        np.testing.assert_array_equal(X, original)
        self.assertTrue(np.all(np.isfinite(gaussian)))

    def test_binary_noise_rejects_nonbinary_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 0 or 1"):
            flip_noise_to_features(
                np.asarray([[0.0, 0.5], [1.0, 0.2]]),
                [1],
                random_state=1,
            )

    def test_mask_is_non_mutating(self) -> None:
        X = np.arange(12).reshape(4, 3)
        masked = mask_features(X, [1], mask_value=-1)
        self.assertTrue(np.all(masked[:, 1] == -1))
        self.assertFalse(np.any(X[:, 1] == -1))

    def test_enumeration_has_record_major_order_and_memory_guard(self) -> None:
        X = np.asarray([[9, 8, 7], [6, 5, 4]], dtype=np.int64)
        result = enumerate_feature_values(
            X,
            [0, 1],
            [0, 2],
            max_output_bytes=10_000,
        )
        self.assertEqual(result.variants_per_record, 4)
        self.assertEqual(result.X.shape, (8, 3))
        np.testing.assert_array_equal(result.X[:4, 1], np.full(4, 8))
        np.testing.assert_array_equal(result.X[4:, 1], np.full(4, 5))

        with self.assertRaisesRegex(MemoryError, "refused"):
            enumerate_feature_values(
                np.ones((100, 20), dtype=np.float64),
                [0, 1],
                list(range(10)),
                max_output_bytes=1000,
            )

    def test_modified_product_is_reproducible(self) -> None:
        X = np.arange(24).reshape(4, 6)
        first = modified_product_dataset(
            X,
            [0, 1],
            0.34,
            random_state=10,
            max_output_bytes=1_000_000,
        )
        second = modified_product_dataset(
            X,
            [0, 1],
            0.34,
            random_state=10,
            max_output_bytes=1_000_000,
        )
        np.testing.assert_array_equal(first, second)

    def test_closest_match_raises_when_no_match_exists(self) -> None:
        X = np.asarray([[0, 0, 0], [1, 1, 1]])
        with self.assertRaisesRegex(LookupError, "No row matches"):
            closest_match_distance(X, np.asarray([0, 1, 0]), [0])


class MetricTests(unittest.TestCase):
    def test_known_confusion_matrix_and_rank_metrics(self) -> None:
        metrics = compute_membership_metrics(
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0.9, 0.4, 0.7, 0.1],
        )
        self.assertEqual(metrics.true_positives, 1)
        self.assertEqual(metrics.false_positives, 1)
        self.assertEqual(metrics.true_negatives, 1)
        self.assertEqual(metrics.false_negatives, 1)
        self.assertAlmostEqual(metrics.accuracy, 0.5)
        self.assertAlmostEqual(metrics.precision, 0.5)
        self.assertAlmostEqual(metrics.recall, 0.5)
        self.assertAlmostEqual(metrics.roc_auc, 0.75)
        self.assertAlmostEqual(metrics.average_precision, (1.0 + 2.0 / 3.0) / 2.0)
        self.assertAlmostEqual(metrics.tpr_at_fpr_1_percent, 0.5)

    def test_undefined_precision_fails(self) -> None:
        with self.assertRaisesRegex(UndefinedMetricError, "precision is undefined"):
            compute_membership_metrics(
                [1, 1, 0, 0],
                [0, 0, 0, 0],
                [0.4, 0.3, 0.2, 0.1],
            )

    def test_rank_metrics_handle_tied_scores_without_order_bias(self) -> None:
        metrics = compute_membership_metrics(
            [1, 0],
            [1, 1],
            [0.5, 0.5],
        )
        self.assertAlmostEqual(metrics.roc_auc, 0.5)
        self.assertAlmostEqual(metrics.average_precision, 0.5)


class ExperimentConfigurationTests(unittest.TestCase):
    def test_smoke_preset_is_explicit_and_json_serializable(self) -> None:
        settings = preset_settings("purchase", "lr", smoke=True)
        self.assertTrue(settings.smoke)
        self.assertEqual(settings.dataset_limit, 5000)
        self.assertEqual(settings.training_size, 500)
        self.assertEqual(settings.testing_size, 200)
        json.dumps(asdict(settings), allow_nan=False)

    def test_rejects_unsupported_enumeration_modality(self) -> None:
        with self.assertRaisesRegex(ValueError, "only defined for binary Purchase"):
            ExperimentSettings(
                modality="newsgroups",
                model_family="lr",
                training_size=100,
                testing_size=50,
                methods=("baseline", "enumerate"),
            )


if __name__ == "__main__":
    unittest.main()
