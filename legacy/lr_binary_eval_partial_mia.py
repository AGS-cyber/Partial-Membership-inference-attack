from sklearn.linear_model import LogisticRegression
from mia import *
from partial_mia import *
import random
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

# TODO: Test with other classifiers. Currently we only use RF for everything.

rng = np.random.default_rng()
# Make sure RF is registered as a classifier with posteriors.
class LR(LogisticRegression, ClassifierWithPosteriors):
    pass


features = np.load('datasets/purchase/modified_purchase100.npy', allow_pickle=True)
labels = np.load('datasets/purchase/cluster_labels_2.npy', allow_pickle=True)

features_list = features.tolist()
labels_list = labels.tolist()

# Create a random dataset.
dataset = LabeledDataset(
  features_list['features'], labels_list
)

# Initialise a sensible MIA config.
config = MIAConfig.from_proportions(dataset.len(), 0.5, 2500, 499)

# Initialise the MIA environment.
env = MIAEnvironment(
    dataset,
    LR(random_state=0),
    LR(random_state=0),
    2,
    LR(random_state=0),
    config,
)

print("Training basic models")
env.train_basic_models()

print("Testing basic models")
print(env.test_basic_models())

print("Training attack models")
env.train_attack_models()

print("Testing attack models")
print(env.test_attack_models())

print("Test evaluation method")
test_size = 2500
print("Results for training data")
available_test_samples = len(env.target_data.testing_data.X)
training_samples = env.target_data.training_data.X[:test_size]
print(env.is_member(training_samples))
train_results, _ = env.is_member(training_samples)
train_member_count = np.sum(train_results)
train_non_member_count = test_size - train_member_count
train_accuracy = train_member_count / test_size
print(f"Number of train samples classified as members: {train_member_count}/{test_size}")

print("Results for testing data")
testing_samples = env.target_data.testing_data.X[:test_size]
print(env.is_member(testing_samples))
test_results, _ = env.is_member(testing_samples)
test_member_count = np.sum(test_results)
test_non_member_count = available_test_samples - test_member_count
test_accuracy = test_member_count / available_test_samples
print(f"Number of test samples classified as members: {test_member_count}/{available_test_samples}")

# Use the same number of train and test data points as the available test samples for calculations
train_size_for_calc = min(test_size, available_test_samples)
train_results_for_calc = train_results[:train_size_for_calc]
train_member_count_for_calc = np.sum(train_results_for_calc)
train_non_member_count_for_calc = train_size_for_calc - train_member_count_for_calc

total_correct = train_member_count_for_calc + test_non_member_count
total_samples = train_size_for_calc + available_test_samples
accuracy = total_correct / total_samples

true_positives = train_member_count_for_calc
false_positives = test_member_count
false_negatives = train_non_member_count_for_calc

precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)

print(f"Overall accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")




print('-' *100)
print("Partial Noise")

print("Results for training data")
available_test_samples1 = len(env.target_data.testing_data.X)
training_samples1 = env.target_data.training_data.X[:test_size]
total_columns_train1 = training_samples1.shape[1]

"--------------------------------------"
percentage_of_columns = 0.05 #adjust this to decide the percentage of columns to add noise on
noise_level = 0.1 #adjust this to adjust noise level 0.1(mild)/0.5(intermediate)/1.0(severe)
"--------------------------------------"

# Calculate the number of columns to which noise will be added
num_columns_to_add_noise = int(total_columns_train1 * percentage_of_columns)

# Randomly select column indices to which noise will be added
column_numbers_to_add_noise_train = np.random.choice(total_columns_train1, size=num_columns_to_add_noise, replace=False)

# Assuming the existence of an add_noise_to_features function that adds noise to specified columns of a dataset
noisy_training_samples = flip_noise_to_features(training_samples1, column_numbers_to_add_noise_train, noise_level)

# Evaluate the MIA on the dataset with added noise
is_member, membership_proba = env.is_member(noisy_training_samples)
print("Membership decisions:")
print(is_member)
print("Membership probabilities:")
print(membership_proba)
train_results1, _ = env.is_member(training_samples)
train_member_count1 = np.sum(train_results1)
train_non_member_count1 = test_size - train_member_count1
train_accuracy1 = train_member_count1 / test_size
print(f"Number of train samples classified as members: {train_member_count1}/{test_size}")

print("Results for testing data")
testing_samples1 = env.target_data.testing_data.X[:test_size]
total_columns_test1 = testing_samples1.shape[1]

# Calculate the number of columns to which noise will be added
num_columns_to_add_noise1 = int(total_columns_test1 * percentage_of_columns)

# Randomly select column indices to which noise will be added
column_numbers_to_add_noise_test = np.random.choice (total_columns_test1, size=num_columns_to_add_noise1, replace=False)

# Assuming the existence of an add_noise_to_features function that adds noise to specified columns of a dataset
noisy_testing_samples = flip_noise_to_features(testing_samples1, column_numbers_to_add_noise_test, noise_level)

# Evaluate the MIA on the dataset with added noise
is_member, membership_proba = env.is_member(noisy_testing_samples)
print("Membership decisions:")
print(is_member)
print("Membership probabilities:")
print(membership_proba)
test_results1, _ = env.is_member(noisy_testing_samples)
test_member_count1 = np.sum(test_results1)
test_non_member_count1 = available_test_samples1 - test_member_count1
test_accuracy1 = test_member_count1 / available_test_samples1
print(f"Number of test samples classified as members: {test_member_count1}/{available_test_samples1}")

# Use the same number of train and test data points as the available test samples for calculations
train_size_for_calc1 = min(test_size, available_test_samples1)
train_results_for_calc1 = train_results1[:train_size_for_calc1]
train_member_count_for_calc1 = np.sum(train_results_for_calc1)
train_non_member_count_for_calc1 = train_size_for_calc1 - train_member_count_for_calc1

total_correct1 = train_member_count_for_calc1 + test_non_member_count1
total_samples1 = train_size_for_calc1 + available_test_samples1
accuracy1 = total_correct1 / total_samples1

true_positives1 = train_member_count_for_calc1
false_positives1 = test_member_count1
false_negatives1 = train_non_member_count_for_calc1

precision1 = true_positives1 / (true_positives1 + false_positives1)
recall1 = true_positives1 / (true_positives1+ false_negatives1)

print(f"Overall accuracy: {accuracy1:.2f}")
print(f"Precision: {precision1:.2f}")
print(f"Recall: {recall1:.2f}")







print('-' *100)
print("Percentage Features")

print("Results for training data")
available_test_samples2 = len(env.target_data.testing_data.X)
training_samples2 = env.target_data.training_data.X[:test_size]
num_features = training_samples2.shape[1]  # Based on the model's expectation

"--------------------------------------"
percentage_to_zero_out = 0.05 #adjust the percentage of feature treat as missing feaures
"--------------------------------------"

# Calculate the number of features to zero out
num_features_to_zero_out = int(num_features * percentage_to_zero_out)

# Randomly select feature indices to zero out
random_indices = rng.choice(num_features, size=num_features_to_zero_out, replace=False)

# Create a mask for all features, then apply the mask to zero-out selected features
mask = np.ones(num_features, dtype=bool)
mask[random_indices] = False  # Set the randomly selected features to zero

# Apply the mask to zero-out the selected features
train_validation_data = np.copy(training_samples2)
train_validation_data[:, ~mask] = -1

# Now call is_member with the full-sized but modified feature dataset
train_is_member2, train_membership_proba2 = env.is_member(train_validation_data)
print("Membership decisions:")
print(train_is_member2)
print("Membership probabilities:")
print(train_membership_proba2)
train_results2, _ = env.is_member(train_validation_data)
train_member_count2 = np.sum(train_results2)
train_non_member_count2 = test_size - train_member_count2
train_accuracy2 = train_member_count2 / test_size
print(f"Number of train samples classified as members: {train_member_count2}/{test_size}")

print("Results for testing data")
testing_samples2 = env.target_data.testing_data.X[:test_size]
num_features2 = testing_samples2.shape[1]  # Based on the model's expectation
num_features_to_zero_out2 = int(num_features * percentage_to_zero_out)

# Randomly select feature indices to zero out
random_indices = rng.choice(num_features2, size=num_features_to_zero_out2, replace=False)

# Create a mask for all features, then apply the mask to zero-out selected features
mask = np.ones(num_features2, dtype=bool)
mask[random_indices] = False  # Set the randomly selected features to zero

# Apply the mask to zero-out the selected features
test_validation_data = np.copy(testing_samples2)
test_validation_data[:, ~mask] = -1

# Now call is_member with the full-sized but modified feature dataset
test_is_member2, test_membership_proba2 = env.is_member(test_validation_data)
print("Membership decisions:")
print(test_is_member2)
print("Membership probabilities:")
print(test_membership_proba2)
test_results2, _ = env.is_member(test_validation_data)
test_member_count2 = np.sum(test_results2)
test_non_member_count2 = available_test_samples2 - test_member_count2
test_accuracy2 = test_member_count2 / available_test_samples2
print(f"Number of test samples classified as members: {test_member_count2}/{available_test_samples2}")

# Use the same number of train and test data points as the available test samples for calculations
train_size_for_calc2 = min(test_size, available_test_samples2)
train_results_for_calc2 = train_results2[:train_size_for_calc2]
train_member_count_for_calc2 = np.sum(train_results_for_calc2)
train_non_member_count_for_calc2 = train_size_for_calc2 - train_member_count_for_calc2

total_correct2 = train_member_count_for_calc2 + test_non_member_count2
total_samples2 = train_size_for_calc2 + available_test_samples2
accuracy2 = total_correct2 / total_samples2

true_positives2 = train_member_count_for_calc2
false_positives2 = test_member_count2
false_negatives2 = train_non_member_count_for_calc2

precision2 = true_positives2 / (true_positives2 + false_positives2)
recall2 = true_positives2 / (true_positives2+ false_negatives2)

print(f"Overall accuracy: {accuracy2:.2f}")
print(f"Precision: {precision2:.2f}")
print(f"Recall: {recall2:.2f}")









print("Feature Selection")
print("Results for training data")
available_test_samples3 = len(env.target_data.testing_data.X)
training_samples3 = env.target_data.training_data.X[:test_size]
labels = env.target_data.training_data.y[:test_size]
num_features = training_samples3.shape[1]  # Based on the model's expectation

"--------------------------------------"
percentage_of_features_to_keep = 0.95  # Adjust this value to keep more or fewer features
"--------------------------------------"

# Calculate the number of features to keep
num_features_to_keep = int(num_features * percentage_of_features_to_keep)

# Select features based on mutual information
mutual_info = mutual_info_classif(training_samples3, labels)
sorted_indices = np.argsort(mutual_info)  # Sort in ascending order of mutual information

# Select the indices of the least mutual features
selected_indices = sorted_indices[:num_features - num_features_to_keep]

# Create a new dataset with the selected features masked with 0
selected_features = np.copy(training_samples3)
selected_features[:, selected_indices] = -1 # Mask the least mutual features with 0

# Reshape the selected_features array to 2D if it has only one sample or one feature
if len(selected_features.shape) == 1 or selected_features.shape[1] == 1:
    selected_features = selected_features.reshape(1, -1)

# Call is_member with the dataset containing the selected features with least mutual information masked
is_member, membership_proba = env.is_member(selected_features)
print("Membership decisions with selected features:")
print(is_member)
print("Membership probabilities with selected features:")
print(membership_proba)
train_results3, _ = env.is_member(selected_features)
train_member_count3 = np.sum(train_results3)
train_non_member_count3 = test_size - train_member_count3
train_accuracy3 = train_member_count3 / test_size
print(f"Number of train samples classified as members: {train_member_count3}/{test_size}")


print("Results for testing data")
testing_samples2 = env.target_data.testing_data.X[:test_size]
labels1 = env.target_data.testing_data.y[:test_size]
num_features2 = testing_samples2.shape[1]  # Based on the model's expectation
num_features_to_keep1 = int(num_features2 * percentage_of_features_to_keep)

# Select features based on mutual information
mutual_info1 = mutual_info_classif(testing_samples2, labels1)
sorted_indices1 = np.argsort(mutual_info1)  # Sort in ascending order of mutual information

# Select the indices of the least mutual features
selected_indices1 = sorted_indices1[:num_features2 - num_features_to_keep1]

# Create a new dataset with the selected features masked with 0
selected_features2 = np.copy(testing_samples2)
selected_features2[:, selected_indices1] = -1  # Mask the least mutual features with 0

# Reshape the selected_features array to 2D if it has only one sample or one feature
if len(selected_features2.shape) == 1 or selected_features2.shape[1] == 1:
    selected_features2 = selected_features2.reshape(1, -1)

# Call is_member with the dataset containing the selected features with least mutual information masked
is_member4, membership_proba4 = env.is_member(selected_features2)
print("Membership decisions with selected features:")
print(is_member4)
print("Membership probabilities with selected features:")
print(membership_proba4)
test_results3, _ = env.is_member(selected_features2)
test_member_count3 = np.sum(test_results3)
test_non_member_count3 = available_test_samples3 - test_member_count3
test_accuracy3 = test_member_count3 / available_test_samples3
print(f"Number of test samples classified as members: {test_member_count3}/{available_test_samples3}")

# Use the same number of train and test data points as the available test samples for calculations
train_size_for_calc3 = min(test_size, available_test_samples3)
train_results_for_calc3 = train_results3[:train_size_for_calc3]
train_member_count_for_calc3 = np.sum(train_results_for_calc3)
train_non_member_count_for_calc3 = train_size_for_calc3 - train_member_count_for_calc3

total_correct3 = train_member_count_for_calc3 + test_non_member_count3
total_samples3 = train_size_for_calc3 + available_test_samples3
accuracy3 = total_correct3 / total_samples3

true_positives3 = train_member_count_for_calc3
false_positives3 = test_member_count3
false_negatives3 = train_non_member_count_for_calc3

precision3 = true_positives3 / (true_positives3 + false_positives3)
recall3 = true_positives3 / (true_positives3 + false_negatives3)

print(f"Overall accuracy: {accuracy3:.2f}")
print(f"Precision: {precision3:.2f}")
print(f"Recall: {recall3:.2f}")







print('-' * 100)
print("Enumerate Features")
r = np.array([0, 1])

"--------------------------------------"
column_percentage = 0.01 # this is the percentage of features that would be treat as missing, but since the computation complexity of this approach, the maximium of my 32gb ram pc could only run with 1 percent of feature, adjust with caution
"--------------------------------------"

print("Results for training data")
training_samples = env.target_data.training_data.X[:test_size]
training_product_dataset, original_columns = modified_product_dataset(
    training_samples, r, column_percentage, return_original_columns=True
)
train_is_member, train_membership_proba = env.is_member(training_product_dataset)
train_member_count = np.sum(train_is_member)
train_non_member_count = training_product_dataset.shape[0] - train_member_count
print(f"Number of train samples classified as members: {train_member_count}/{training_product_dataset.shape[0]}")

print("Results for non-training data")
testing_samples = env.target_data.testing_data.X[:test_size]
testing_product_dataset, original_columns = modified_product_dataset(
    testing_samples, r, column_percentage, return_original_columns=True
)
test_is_member, test_membership_proba = env.is_member(testing_product_dataset)
test_member_count = np.sum(test_is_member)
test_non_member_count = testing_product_dataset.shape[0] - test_member_count
print(f"Number of test samples classified as members: {test_member_count}/{testing_product_dataset.shape[0]}")

# Calculate accuracy, precision, and recall
total_samples = training_product_dataset.shape[0] + testing_product_dataset.shape[0]
total_correct = train_member_count + test_non_member_count
accuracy = total_correct / total_samples

true_positives = train_member_count
false_positives = test_member_count
false_negatives = training_product_dataset.shape[0] - train_member_count

precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)

print(f"Overall accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")



