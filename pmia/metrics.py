"""Strict membership-inference metrics with explicit undefined failures."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


class UndefinedMetricError(ValueError):
    """Raised when a requested metric has no mathematically valid denominator."""


def _binary_vector(values: ArrayLike, *, name: str) -> NDArray[np.int64]:
    vector = np.asarray(values)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector")
    if vector.dtype == np.bool_:
        vector = vector.astype(np.int64)
    if not np.all((vector == 0) | (vector == 1)):
        raise ValueError(f"{name} must contain only binary values 0 and 1")
    return vector.astype(np.int64, copy=False)


def _probability_vector(values: ArrayLike, *, expected_size: int) -> NDArray[np.float64]:
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.ndim != 1 or probabilities.size != expected_size:
        raise ValueError(
            "membership_probabilities must be one-dimensional and match y_true; "
            f"got shape {probabilities.shape}, expected ({expected_size},)"
        )
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("membership_probabilities contain NaN or infinity")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("membership_probabilities must be in [0, 1]")
    return probabilities


def _divide(numerator: int | float, denominator: int | float, *, name: str) -> float:
    if denominator == 0:
        raise UndefinedMetricError(
            f"{name} is undefined because its denominator is zero"
        )
    value = float(numerator / denominator)
    if not np.isfinite(value):
        raise UndefinedMetricError(f"{name} is not finite")
    return value


def _average_tie_ranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def roc_auc(
    y_true: NDArray[np.int64],
    probabilities: NDArray[np.float64],
) -> float:
    positive_count = int(np.sum(y_true == 1))
    negative_count = int(np.sum(y_true == 0))
    if positive_count == 0 or negative_count == 0:
        raise UndefinedMetricError("ROC AUC requires both member and non-member records")
    ranks = _average_tie_ranks(probabilities)
    positive_rank_sum = float(ranks[y_true == 1].sum())
    auc = (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)
    if not 0.0 <= auc <= 1.0:
        raise RuntimeError(f"Calculated ROC AUC is outside [0, 1]: {auc}")
    return float(auc)


def average_precision(
    y_true: NDArray[np.int64],
    probabilities: NDArray[np.float64],
) -> float:
    positive_count = int(np.sum(y_true == 1))
    if positive_count == 0:
        raise UndefinedMetricError("Average precision requires member records")
    order = np.argsort(-probabilities, kind="mergesort")
    sorted_labels = y_true[order]
    sorted_probabilities = probabilities[order]
    cumulative_positives = np.cumsum(sorted_labels == 1)
    cumulative_total = np.arange(1, len(sorted_labels) + 1)
    threshold_ends = np.flatnonzero(
        np.r_[sorted_probabilities[1:] != sorted_probabilities[:-1], True]
    )
    true_positives_at_threshold = cumulative_positives[threshold_ends]
    precision_at_threshold = (
        true_positives_at_threshold / cumulative_total[threshold_ends]
    )
    new_positives = np.diff(np.r_[0, true_positives_at_threshold])
    recall_increments = new_positives / positive_count
    result = float(np.sum(recall_increments * precision_at_threshold))
    if not 0.0 <= result <= 1.0:
        raise RuntimeError(
            f"Calculated average precision is outside [0, 1]: {result}"
        )
    return result


def tpr_at_max_fpr(
    y_true: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    max_fpr: float,
) -> float:
    if not isinstance(max_fpr, (int, float, np.floating)):
        raise TypeError("max_fpr must be numeric")
    max_fpr = float(max_fpr)
    if not 0.0 <= max_fpr <= 1.0:
        raise ValueError("max_fpr must be in [0, 1]")

    positive_count = int(np.sum(y_true == 1))
    negative_count = int(np.sum(y_true == 0))
    if positive_count == 0 or negative_count == 0:
        raise UndefinedMetricError("TPR at FPR requires both binary classes")

    order = np.argsort(-probabilities, kind="mergesort")
    labels = y_true[order]
    scores = probabilities[order]
    cumulative_true_positives = np.cumsum(labels == 1)
    cumulative_false_positives = np.cumsum(labels == 0)
    threshold_ends = np.flatnonzero(
        np.r_[scores[1:] != scores[:-1], True]
    )
    tpr = cumulative_true_positives[threshold_ends] / positive_count
    fpr = cumulative_false_positives[threshold_ends] / negative_count
    valid_tpr = tpr[fpr <= max_fpr]
    return float(np.max(valid_tpr)) if valid_tpr.size else 0.0


@dataclass(frozen=True, slots=True)
class MembershipMetrics:
    total: int
    members: int
    non_members: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    accuracy: float
    precision: float
    recall: float
    specificity: float
    false_positive_rate: float
    attack_advantage: float
    roc_auc: float
    average_precision: float
    tpr_at_fpr_1_percent: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_membership_metrics(
    y_true: ArrayLike,
    decisions: ArrayLike,
    membership_probabilities: ArrayLike,
) -> MembershipMetrics:
    truth = _binary_vector(y_true, name="y_true")
    predicted = _binary_vector(decisions, name="decisions")
    if predicted.size != truth.size:
        raise ValueError(
            f"decisions has {predicted.size} records; y_true has {truth.size}"
        )
    probabilities = _probability_vector(
        membership_probabilities,
        expected_size=truth.size,
    )

    true_positives = int(np.sum((truth == 1) & (predicted == 1)))
    false_positives = int(np.sum((truth == 0) & (predicted == 1)))
    true_negatives = int(np.sum((truth == 0) & (predicted == 0)))
    false_negatives = int(np.sum((truth == 1) & (predicted == 0)))
    members = true_positives + false_negatives
    non_members = true_negatives + false_positives

    recall = _divide(
        true_positives,
        true_positives + false_negatives,
        name="recall",
    )
    false_positive_rate = _divide(
        false_positives,
        false_positives + true_negatives,
        name="false_positive_rate",
    )
    return MembershipMetrics(
        total=int(truth.size),
        members=members,
        non_members=non_members,
        true_positives=true_positives,
        false_positives=false_positives,
        true_negatives=true_negatives,
        false_negatives=false_negatives,
        accuracy=_divide(
            true_positives + true_negatives,
            truth.size,
            name="accuracy",
        ),
        precision=_divide(
            true_positives,
            true_positives + false_positives,
            name="precision",
        ),
        recall=recall,
        specificity=_divide(
            true_negatives,
            true_negatives + false_positives,
            name="specificity",
        ),
        false_positive_rate=false_positive_rate,
        attack_advantage=recall - false_positive_rate,
        roc_auc=roc_auc(truth, probabilities),
        average_precision=average_precision(truth, probabilities),
        tpr_at_fpr_1_percent=tpr_at_max_fpr(
            truth,
            probabilities,
            max_fpr=0.01,
        ),
    )
