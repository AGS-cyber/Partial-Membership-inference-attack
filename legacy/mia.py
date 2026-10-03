from dataclasses import dataclass
from typing import List
from datastructures import (
    ClassifierList,
    ClassifierWithPosteriors,
    LabeledDataset,
    TrainTestSplit,
)
from setup import *
import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing_extensions import Self


@dataclass
class MIAConfig:
    """
    The full dataset is split into multiple subsets.
    1. It will be split into a dataset used for the target models and a dataset used for the shadow models.
       The first `absolute_target_size` elements are allocated to the target models, the remainder to the shadow models.
    2. Each of those datasets is then split into a training and testing dataset according to the `training_proportion`.
    3. From these sets, a separate training and testing set is sampled of the respective sizes
       `absolute_training_size` and `absolute_testing_size`.
    """

    absolute_target_size: int  # The absolute number of samples being allocated to the target models.
    absolute_training_size: int  # The absolute number of training samples used to train a single classifier.
    absolute_testing_size: int  # The absolute number of training samples used to test a single classifier.
    training_proportion: float = (
        0.7  # The proportion of samples allocated for training.
    )

    def __post_init__(self):
        assert (
            0 < self.training_proportion < 1
        ), "training_proportion needs to be between 0 and 1"

    def validate(self, num_samples: int):
        assert (
            num_samples > self.absolute_target_size
        ), "Number of samples must be larger than the samples used for the target models"

        # Calculation for target models
        maximum_training_size = int(
            self.training_proportion * self.absolute_target_size
        )
        assert (
            self.absolute_training_size <= maximum_training_size
        ), "Dataset/proportions do not allow for training sets larger than {}".format(
            maximum_training_size
        )
        maximum_testing_size = int(
            (1 - self.training_proportion) * self.absolute_target_size
        )
        assert (
            self.absolute_testing_size <= maximum_testing_size
        ), "Dataset/proportions do not allow for testing sets larger than {}".format(
            maximum_testing_size
        )

        # Calculation for shadow models
        absolute_shadow_size = num_samples - self.absolute_target_size
        maximum_training_size = int(self.training_proportion * absolute_shadow_size)
        assert (
            self.absolute_training_size <= maximum_training_size
        ), "Dataset/proportions do not allow for training sets larger than {}".format(
            maximum_training_size
        )
        maximum_testing_size = int(
            (1 - self.training_proportion) * absolute_shadow_size
        )
        assert (
            self.absolute_testing_size <= maximum_testing_size
        ), "Dataset/proportions do not allow for testing sets larger than {}".format(
            maximum_testing_size
        )

    @staticmethod
    def from_proportions(
        num_samples: int,
        relative_target_size: float,
        absolute_training_size: int,
        absolute_testing_size: int,
        training_proportion: float = 0.7,
    ) -> Self:
        """
        This method calculates the absolute MIA config from a relative target size and a dataset size.
        It also fails early should the required training/testing sizes not be met.
        """
        assert (
            0 < relative_target_size < 1
        ), "relative_target_size needs to be between 0 and 1"

        # Calculation for target models
        absolute_target_size = int(num_samples * relative_target_size)
        maximum_training_size = int(training_proportion * absolute_target_size)
        assert (
            absolute_training_size <= maximum_training_size
        ), "Dataset/proportions do not allow for training sets larger than {}".format(
            maximum_training_size
        )
        maximum_testing_size = int((1 - training_proportion) * absolute_target_size)
        assert (
            absolute_testing_size <= maximum_testing_size
        ), "Dataset/proportions do not allow for testing sets larger than {}".format(
            maximum_testing_size
        )

        return MIAConfig(
            absolute_target_size,
            absolute_training_size,
            absolute_testing_size,
            training_proportion,
        )


class MIAEnvironment:
    def __init__(
        self,
        dataset: LabeledDataset,
        target_model: ClassifierWithPosteriors,
        shadow_model_base: ClassifierWithPosteriors,
        num_shadow_models: int,
        attack_model_base: ClassifierWithPosteriors,
        config: MIAConfig,
        random_state: int = 0,
    ) -> None:
        """
        Create an evaluation environment for a membership inference attack.
        The MIA receives a labeled dataset and a list of target and shadow models to be trained with this data.
        This constructor will then split the dataset accordingly but will not run any training/testing.
        """
        # Make sure dataset only uses numeric, ascending labels.
        self.label_encoder = dataset.transform_label()
        self.num_classes = len(self.label_encoder.classes_)

        # Clone models.
        # We want to be able to have multiple shadow models.
        # We want one attack model per class.
        shadow_models = ClassifierList(
            [shadow_model_base.clone() for _ in range(num_shadow_models)]
        )
        attack_models = ClassifierList(
            [attack_model_base.clone() for _ in range(self.num_classes)]
        )

        # Validate config first to avoid disappointment later.
        config.validate(dataset.len())

        # Split dataset into target and shadow parts.
        target_data, shadow_data = dataset.split_at(config.absolute_target_size)

        # Split target/shadow datasets into training/testing parts.
        target_data = target_data.split(
            training_proportion=config.training_proportion, random_state=random_state
        )
        shadow_data = shadow_data.split(
            training_proportion=config.training_proportion, random_state=random_state
        )

        # Resample each to get datasets for each model to be trained.
        target_data = target_data.sample(
            config.absolute_training_size,
            config.absolute_testing_size,
            random_state=random_state,
        )
        shadow_data = shadow_data.sample_n(
            config.absolute_training_size,
            config.absolute_testing_size,
            len(shadow_models),
            random_state=random_state,
        )

        # Set datasets.
        self.target_data = target_data
        self.shadow_data = shadow_data
        self.attack_data = None

        # Set classifiers.
        self.target_model = target_model
        self.shadow_models = shadow_models
        self.attack_models = attack_models

        # Set flags.
        self.trained_basic_models = False
        self.trained_attack_models = False
        self.config = config

    def train_basic_models(self):
        """
        This method will train the target and shadow models.
        """
        self.target_model.train(self.target_data.training_data)
        self.shadow_models.train(self.shadow_data)
        self.trained_basic_models = True

    def test_basic_models(self) -> (float, List[float]):
        """
        This method will test the target model and shadow models and return the corresponding results.
        """
        assert (
            self.trained_basic_models
        ), "Target and shadow models need to be trained first."

        target_result = self.target_model.test(self.target_data.testing_data)
        shadow_result = self.shadow_models.test(self.shadow_data)
        return target_result, shadow_result

    def _get_attack_dataset(
        self,
        models: ClassifierList,
        datasets: List[TrainTestSplit],
        data_size: int,
        random_state: int = 0,
    ):
        """
        Calculates an attack dataset for class from the shadow models.
        """
        # For each class label of the original data, we get a dataset.
        attack_X = [[] for _ in range(self.num_classes)]
        attack_y = [[] for _ in range(self.num_classes)]

        # We use the training/test data from the given models.
        for model_idx in range(len(models)):
            dataset = datasets[model_idx]

            # Sample a smaller dataset if necessary.
            dataset = dataset.sample(data_size, data_size, random_state=random_state)

            # For each sample, predict the probabilities.
            # Training data was used to train the shadow model and thus gets the label IN.
            # Testing data gets the label OUT.
            in_results = models[model_idx].predict_proba(dataset.training_data.X)
            out_results = models[model_idx].predict_proba(dataset.testing_data.X)

            # Attribute data to the right class.
            for sample_idx in range(data_size):
                # Get 0 based label for training sample and add to corresponding attack dataset.
                class_idx = dataset.training_data.y[sample_idx]
                attack_X[class_idx].append(
                    in_results[sample_idx]
                )  # sample_idx's row in the result -> posteriors
                attack_y[class_idx].append(IN)

                # Get 0 based label for testing sample and add to corresponding attack dataset.
                class_idx = dataset.testing_data.y[sample_idx]
                attack_X[class_idx].append(
                    out_results[sample_idx]
                )  # sample_idx's row in the result -> posteriors
                attack_y[class_idx].append(OUT)

        # Convert to labeled datasets.
        return [
            LabeledDataset(attack_X[i], attack_y[i]) for i in range(self.num_classes)
        ]

    def train_attack_models(self, random_state: int = 0):
        """
        This method will train the attack models.
        """
        assert (
            self.trained_basic_models
        ), "Target and shadow models need to be trained first."

        if self.attack_data is None:
            attack_train = self._get_attack_dataset(
                self.shadow_models,
                self.shadow_data,
                min(
                    self.config.absolute_training_size,
                    self.config.absolute_testing_size,
                ),
                random_state=random_state,
            )
            # We only have one target model but the attack dataset is constructed from a list of models for generality.
            attack_test = self._get_attack_dataset(
                ClassifierList([self.target_model]),
                [self.target_data],
                min(
                    self.config.absolute_training_size,
                    self.config.absolute_testing_size,
                ),
                random_state=random_state,
            )
            self.attack_data = [
                TrainTestSplit(attack_train[class_idx], attack_test[class_idx])
                for class_idx in range(len(attack_train))
            ]

        self.attack_models.train(self.attack_data)
        self.trained_attack_models = True

    def test_attack_models(self) -> List[float]:
        """
        This method will test the attack models.
        """
        assert self.trained_attack_models, "Attack models need to be trained first."

        return self.attack_models.test(self.attack_data)

    def is_member(self, X: ArrayLike) -> (NDArray[np.bool_], NDArray[np.float_]):
        """
        This method evaluates whether one or multiple inputs (depending on whether it's a vector or matrix)
        were members of the target model's training set.

        It returns both the boolean decisions and the membership posterior.
        Both return values are 1d vectors (i.e., we only report the posterior for the membership class).
        """
        # First determine classes for X.
        target_posteriors = self.target_model.predict_proba(X)
        class_idxs = np.argmax(target_posteriors, axis=1)

        # Group the posteriors by class indices for faster querying of the attack models.
        # First, sort posteriors and indices by ascending class indices.
        sorting_key = class_idxs.argsort()
        target_posteriors = target_posteriors[sorting_key]
        class_idxs = class_idxs[sorting_key]
        # Second, get indices where splitting is required.
        unique_values, unique_indices = np.unique(class_idxs, return_index=True)
        # The first index is always 0 and doesn't provide any information, skip it.
        unique_indices = unique_indices[1:]
        # Split the posteriors.
        target_posteriors_by_class = [
            np.array([]) for _ in range(len(self.attack_models))
        ]
        split_target_posteriors = np.split(target_posteriors, unique_indices)
        for class_idx, posteriors in zip(unique_values, split_target_posteriors):
            target_posteriors_by_class[class_idx] = posteriors

        # Then we run the corresponding attack models.
        # Remember that we have one attack model per class label.
        is_member_results = np.array([])
        membership_posteriors_results = np.array([])
        for class_idx in range(len(self.attack_models)):
            attack_posteriors = self.attack_models[class_idx].predict_proba(
                target_posteriors_by_class[class_idx]
            )
            is_member = np.argmax(attack_posteriors, axis=1) == IN
            membership_posteriors = attack_posteriors[:, IN]

            is_member_results = np.append(is_member_results, is_member)
            membership_posteriors_results = np.append(
                membership_posteriors_results, membership_posteriors
            )

        return is_member_results, membership_posteriors_results