from dataclasses import dataclass
from numpy.typing import ArrayLike
from typing_extensions import Self
from typing import List
from abc import ABCMeta, abstractmethod
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import numpy as np


@dataclass
class LabeledDataset:
    X: ArrayLike
    y: ArrayLike

    def __post_init__(self):
        assert len(self.X) == len(self.y)

    @staticmethod
    def from_pandas(df: pd.DataFrame, label_column: str) -> Self:
        """
        Creates a labeled dataset from a pandas dataframe.
        """
        y = df[label_column].values
        X = df.drop(label_column, axis=1).values
        return LabeledDataset(X, y)

    def transform_label(self) -> LabelEncoder:
        """
        Transforms labels to numeric labels and returns the encoder.
        The encoder can be used to transform labels back.
        """
        labelencoder = LabelEncoder()
        self.y = labelencoder.fit_transform(self.y)
        return labelencoder

    def len(self) -> int:
        return len(self.y)

    def __len__(self) -> int:
        return self.len()

    def split(
        self, training_proportion: float = 0.8, random_state: int = 0
    ) -> "TrainTestSplit":
        """
        Splits the dataset into training and testing data.
        """
        assert (
            0 < training_proportion < 1
        ), "training_proportion needs to be between 0 and 1"

        X_train, X_test, y_train, y_test = train_test_split(
            self.X,
            self.y,
            train_size=training_proportion,
            random_state=random_state,
            stratify=self.y,
        )
        return TrainTestSplit(
            LabeledDataset(X_train, y_train), LabeledDataset(X_test, y_test)
        )

    def split_at(self, index: int) -> (Self, Self):
        """
        Splits dataset in two parts at index.
        """
        return (
            LabeledDataset(self.X[:index], self.y[:index]),
            LabeledDataset(self.X[index:], self.y[index:]),
        )

    def sample(self, n_samples: int, random_state: int = 0) -> Self:
        """
        This function samples a new dataset with `n_samples` elements from this one.
        """
        X, y = resample(
            self.X,
            self.y,
            n_samples=n_samples,
            replace=False,
            random_state=random_state,
        )
        return LabeledDataset(X, y)

    def num_classes(self) -> int:
        return len(np.unique(self.y))


@dataclass
class TrainTestSplit:
    training_data: LabeledDataset
    testing_data: LabeledDataset

    def split_at(self, training_index: int, testing_index: int) -> (Self, Self):
        """
        Splits dataset in two parts at index.
        """
        training1, training2 = self.training_data.split_at(training_index)
        testing1, testing2 = self.testing_data.split_at(testing_index)
        return (
            TrainTestSplit(training1, testing1),
            TrainTestSplit(training2, testing2),
        )

    def sample(
        self, training_size: int, testing_size: int, random_state: int = 0
    ) -> Self:
        """
        This function samples a new dataset with `n_samples` elements from this one.
        """
        return TrainTestSplit(
            self.training_data.sample(training_size, random_state=random_state),
            self.testing_data.sample(testing_size, random_state=random_state),
        )

    def sample_n(
        self,
        training_size: int,
        testing_size: int,
        num_sets: int,
        random_state: int = 0,
    ) -> List[Self]:
        """
        This function takes training and test data and resamples `num_sets` separate training/testing datasets.
        It does so with replacement, thus the same training/testing sample may occur in multiple resulting datasets.
        We can use it to create data sets for multiple (shadow/target) models.
        """
        new_datasets = []
        for i in range(num_sets):
            new_datasets.append(
                self.sample(training_size, testing_size, random_state=random_state)
            )
        return new_datasets


class ClassifierWithPosteriors(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        pass

    @abstractmethod
    def predict_proba(self, X: ArrayLike) -> ArrayLike:
        pass

    @abstractmethod
    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        pass

    def train(self, dataset: LabeledDataset) -> Self:
        return self.fit(dataset.X, dataset.y)

    def test(self, dataset: LabeledDataset) -> float:
        return self.score(dataset.X, dataset.y)

    def clone(self) -> Self:
        return sklearn.base.clone(self)


class ClassifierList:
    def __init__(self, models: List[ClassifierWithPosteriors]) -> None:
        self.models = models

    def train(self, dataset: List[TrainTestSplit]):
        assert len(dataset) == len(self.models)
        for i in range(len(self.models)):
            self.models[i].train(dataset[i].training_data)

    def test(self, dataset: List[TrainTestSplit]) -> List[float]:
        assert len(dataset) == len(self.models)
        results = []
        for i in range(len(self.models)):
            results.append(self.models[i].test(dataset[i].testing_data))
        return results

    def len(self) -> int:
        return len(self.models)

    def __len__(self) -> int:
        return self.len()

    def __getitem__(self, key):
        return self.models[key]
