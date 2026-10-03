"""Corrected, isolated implementation of the partial-membership MIA project."""

from .constants import IN, OUT
from .datastructures import ClassifierList, LabeledDataset, TrainTestSplit
from .mia import MIAConfig, MIAEnvironment

__all__ = [
    "IN",
    "OUT",
    "ClassifierList",
    "LabeledDataset",
    "TrainTestSplit",
    "MIAConfig",
    "MIAEnvironment",
]

__version__ = "1.0.0"
