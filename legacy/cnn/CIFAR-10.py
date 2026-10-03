import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers, models, optimizers

# Load and preprocess the CIFAR-100 dataset
(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train_full = X_train_full.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Use a subset of the CIFAR-10 training data
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# Target model
def create_target_model(learning_rate=0.001):
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.5),
        
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

target_learning_rate= 0.001
target_model = create_target_model(learning_rate=target_learning_rate)
history = target_model.fit(X_train, y_train, epochs=100, batch_size=256, 
                           validation_data=(X_val, y_val), verbose=1)

print(f"Target Model - Validation Accuracy: {target_model.evaluate(X_val, y_val, verbose=1)[1]:.4f}")

# Function to add Gaussian noise
def add_gaussian_noise(X, std_dev):
    return np.clip(X + np.random.normal(0, std_dev, X.shape), 0, 1)

# Shadow models
num_shadow_models = 20
shadow_models = []

for i in range(num_shadow_models):
    print(f"Training shadow model {i+1}/{num_shadow_models}")
    X_shadow, X_shadow_out, y_shadow, y_shadow_out = train_test_split(X_test, y_test, test_size=0.5, random_state=i)
    
    shadow_model = create_target_model()
    shadow_model.fit(X_shadow, y_shadow, epochs=100, batch_size=256, 
                     validation_split=0.2, verbose=1)
    shadow_models.append(shadow_model)

# Prepare data for attack models
def prepare_attack_data(model, X, y, is_member):
    predictions = model.predict(X)
    labels = np.eye(10)[y.reshape(-1)]
    membership = np.full(len(X), is_member)
    return np.hstack((predictions, labels, membership.reshape(-1, 1)))

attack_data = []
for shadow_model in shadow_models:
    X_shadow, X_shadow_out, y_shadow, y_shadow_out = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    attack_data.append(prepare_attack_data(shadow_model, X_shadow, y_shadow, 1))
    attack_data.append(prepare_attack_data(shadow_model, X_shadow_out, y_shadow_out, 0))

attack_data = np.vstack(attack_data)

# Train attack models (one per class)
def create_attack_model(learning_rate=0.25):
    model = models.Sequential([
        layers.Dense(256, activation='relu', input_shape=(200,)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=0.25),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

attack_learning_rate= 0.25
attack_models = []
for class_idx in range(10):
    print(f"Training attack model for class {class_idx+1}/100")
    class_data = attack_data[attack_data[:, 100:200].argmax(axis=1) == class_idx]
    X_attack = class_data[:, :200]  # Use both prediction probabilities and true labels
    y_attack = class_data[:, -1]  # Membership label
    
    attack_model = create_attack_model(learning_rate=attack_learning_rate)
    attack_model.fit(X_attack, y_attack, epochs=200, batch_size=32, 
                     validation_split=0.2, verbose=1)
    attack_models.append(attack_model)

# Evaluate the attack
def evaluate_attack(target_model, attack_models, X, y, is_member, noise_level):
    X_noisy = add_gaussian_noise(X, noise_level)
    target_predictions = target_model.predict(X_noisy)
    labels = np.eye(10)[y.reshape(-1)]
    attack_input = np.hstack((target_predictions, labels))
    memberships = []
    
    for class_idx in range(10):
        class_mask = y.reshape(-1) == class_idx
        if np.sum(class_mask) > 0:
            attack_predictions = attack_models[class_idx].predict(attack_input[class_mask])
            memberships.extend(attack_predictions.flatten())
    
    predicted_memberships = (np.array(memberships) > 0.5).astype(int)
    true_memberships = np.full(len(memberships), is_member)
    
    if is_member:  # Training data
        true_positives = np.sum(predicted_memberships)
        false_negatives = np.sum(1 - predicted_memberships)
    else:  # Testing data
        false_positives = np.sum(predicted_memberships)
        true_negatives = np.sum(1 - predicted_memberships)
    
    accuracy = accuracy_score(true_memberships, predicted_memberships)
    
    return {
        'total_samples': len(memberships),
        'predicted_members': np.sum(predicted_memberships),
        'accuracy': accuracy,
        'true_positives': true_positives if is_member else 0,
        'false_negatives': false_negatives if is_member else 0,
        'false_positives': false_positives if not is_member else 0,
        'true_negatives': true_negatives if not is_member else 0
    }

# Evaluate under various noise conditions
noise_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for noise in noise_levels:
    print(f"\nEvaluating with noise level: {noise}")
    
    train_results = evaluate_attack(target_model, attack_models, X_train, y_train, True, noise)
    test_results = evaluate_attack(target_model, attack_models, X_test, y_test, False, noise)
    
    print("Training Data Results:")
    print(f"  True Positives (Correctly identified as members): {train_results['true_positives']} out of {train_results['total_samples']}")
    print(f"  False Negatives (Incorrectly identified as non-members): {train_results['false_negatives']}")
    print(f"  Accuracy: {train_results['accuracy']:.4f}")
    
    print("\nTesting Data Results:")
    print(f"  False Positives (Incorrectly identified as members): {test_results['false_positives']} out of {test_results['total_samples']}")
    print(f"  True Negatives (Correctly identified as non-members): {test_results['true_negatives']}")
    print(f"  Accuracy: {test_results['accuracy']:.4f}")
    
    # Calculate overall metrics
    total_samples = train_results['total_samples'] + test_results['total_samples']
    true_positives = train_results['true_positives']
    false_positives = test_results['false_positives']
    true_negatives = test_results['true_negatives']
    false_negatives = train_results['false_negatives']
    
    overall_accuracy = (train_results['accuracy'] * train_results['total_samples'] + 
                        test_results['accuracy'] * test_results['total_samples']) / total_samples
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    print("\nOverall Results:")
    print(f"  Total Samples: {total_samples}")
    print(f"  True Positives: {true_positives}")
    print(f"  False Positives: {false_positives}")
    print(f"  True Negatives: {true_negatives}")
    print(f"  False Negatives: {false_negatives}")
    print(f"  Accuracy: {overall_accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")


