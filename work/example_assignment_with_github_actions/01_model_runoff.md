# Part 1: Model Run-off (Systematic Model Selection)

## Introduction

In this part, you'll implement a systematic approach to model selection for healthcare data. You'll compare multiple model architectures, evaluate their performance, and select the best model based on various metrics.

## Learning Objectives

- Implement a systematic model selection process
- Compare multiple model architectures
- Apply cross-validation techniques
- Analyze performance trade-offs
- Save model and metrics in the correct format

## Setup and Installation

```python
# Install required packages
%pip install -r requirements.txt

# Import necessary libraries
import tensorflow as tf
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Set random seeds for reproducibility
tf.random.set_seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Configure matplotlib for better visualization
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('results/part_1', exist_ok=True)
os.makedirs('logs', exist_ok=True)
```

## 1. Data Loading and Preprocessing

```python
# Load healthcare dataset
# This is a placeholder - replace with your actual dataset loading code
# Example:
# from sklearn.datasets import load_breast_cancer
# data = load_breast_cancer()
# X, y = data.data, data.target

# For demonstration, we'll use a synthetic dataset
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
                          n_redundant=5, n_classes=2, random_state=42)

# Print dataset information
print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Class distribution: {np.bincount(y)}")

# Split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")
print(f"Test set shape: {X_test.shape}")

# Standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

## 2. Model Definitions

```python
# Define multiple model architectures to compare

def create_model_1(input_shape, num_classes):
    """
    Simple neural network with 2 hidden layers
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_model_2(input_shape, num_classes):
    """
    Deeper neural network with 4 hidden layers
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_model_3(input_shape, num_classes):
    """
    Wide neural network with residual connections
    """
    inputs = tf.keras.Input(shape=(input_shape,))
    x = tf.keras.layers.Dense(256, activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Residual block 1
    block_1 = tf.keras.layers.Dense(256, activation='relu')(x)
    block_1 = tf.keras.layers.BatchNormalization()(block_1)
    block_1 = tf.keras.layers.Dropout(0.3)(block_1)
    block_1 = tf.keras.layers.Dense(256, activation='relu')(block_1)
    block_1 = tf.keras.layers.BatchNormalization()(block_1)
    block_1 = tf.keras.layers.Dropout(0.3)(block_1)
    x = tf.keras.layers.add([x, block_1])
    
    # Output layer
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Create model instances
model_1 = create_model_1(X_train.shape[1], len(np.unique(y)))
model_2 = create_model_2(X_train.shape[1], len(np.unique(y)))
model_3 = create_model_3(X_train.shape[1], len(np.unique(y)))

# Print model summaries
print("Model 1 Summary:")
model_1.summary()
print("\nModel 2 Summary:")
model_2.summary()
print("\nModel 3 Summary:")
model_3.summary()
```

## 3. Cross-Validation and Model Selection

```python
# Define a function to evaluate models using cross-validation
def evaluate_model_cv(model_fn, X, y, n_splits=5):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Create and train model
        model = model_fn(X.shape[1], len(np.unique(y)))
        model.fit(
            X_train_fold, y_train_fold,
            epochs=50,
            batch_size=32,
            verbose=0,
            validation_data=(X_val_fold, y_val_fold),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            ]
        )
        
        # Evaluate model
        _, accuracy = model.evaluate(X_val_fold, y_val_fold, verbose=0)
        scores.append(accuracy)
        
        print(f"Fold {fold+1}: Accuracy = {accuracy:.4f}")
    
    return scores

# Evaluate models using cross-validation
print("Evaluating Model 1...")
model_1_scores = evaluate_model_cv(create_model_1, X_train_scaled, y_train)
print(f"Model 1 Mean Accuracy: {np.mean(model_1_scores):.4f}, Std: {np.std(model_1_scores):.4f}")

print("\nEvaluating Model 2...")
model_2_scores = evaluate_model_cv(create_model_2, X_train_scaled, y_train)
print(f"Model 2 Mean Accuracy: {np.mean(model_2_scores):.4f}, Std: {np.std(model_2_scores):.4f}")

print("\nEvaluating Model 3...")
model_3_scores = evaluate_model_cv(create_model_3, X_train_scaled, y_train)
print(f"Model 3 Mean Accuracy: {np.mean(model_3_scores):.4f}, Std: {np.std(model_3_scores):.4f}")

# Compare model performance
model_names = ['Simple NN', 'Deep NN', 'Residual NN']
mean_scores = [np.mean(model_1_scores), np.mean(model_2_scores), np.mean(model_3_scores)]
std_scores = [np.std(model_1_scores), np.std(model_2_scores), np.std(model_3_scores)]

plt.figure(figsize=(10, 6))
plt.bar(model_names, mean_scores, yerr=std_scores, capsize=10)
plt.ylim(0.5, 1.0)
plt.ylabel('Cross-Validation Accuracy')
plt.title('Model Comparison: Cross-Validation Performance')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Select the best model based on cross-validation
best_model_idx = np.argmax(mean_scores)
best_model_name = model_names[best_model_idx]
print(f"Best model based on cross-validation: {best_model_name}")

# Create and train the best model on the full training set
if best_model_idx == 0:
    best_model_fn = create_model_1
elif best_model_idx == 1:
    best_model_fn = create_model_2
else:
    best_model_fn = create_model_3

best_model = best_model_fn(X_train.shape[1], len(np.unique(y)))
```

## 4. Training and Evaluation of Best Model

```python
# Define callbacks for the best model
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'models/best_model.keras',
        save_best_only=True
    )
]

# Train the best model
history = best_model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks
)

# Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot accuracy
ax1.plot(history.history['accuracy'], label='Training')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()

# Plot loss
ax2.plot(history.history['loss'], label='Training')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()

plt.tight_layout()
plt.show()

# Evaluate best model on test set
test_loss, test_accuracy = best_model.evaluate(X_test_scaled, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

# Get predictions
predictions = best_model.predict(X_test_scaled)
predicted_labels = np.argmax(predictions, axis=1)

# Calculate metrics
precision = precision_score(y_test, predicted_labels)
recall = recall_score(y_test, predicted_labels)
f1 = f1_score(y_test, predicted_labels)
cm = confusion_matrix(y_test, predicted_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Save metrics
metrics = {
    'model': best_model_name,
    'accuracy': float(test_accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'confusion_matrix': cm.tolist()
}

# Save to file
with open('results/part_1/model_comparison.txt', 'w') as f:
    f.write(f"best_model: {metrics['model']}\n")
    f.write(f"accuracy: {metrics['accuracy']}\n")
    f.write(f"precision: {metrics['precision']}\n")
    f.write(f"recall: {metrics['recall']}\n")
    f.write(f"f1_score: {metrics['f1_score']}\n")
    f.write(f"confusion_matrix: {metrics['confusion_matrix']}\n")
    
    # Also save cross-validation results
    f.write("\n--- Cross-Validation Results ---\n")
    f.write(f"model_1_mean_accuracy: {np.mean(model_1_scores)}\n")
    f.write(f"model_1_std_accuracy: {np.std(model_1_scores)}\n")
    f.write(f"model_2_mean_accuracy: {np.mean(model_2_scores)}\n")
    f.write(f"model_2_std_accuracy: {np.std(model_2_scores)}\n")
    f.write(f"model_3_mean_accuracy: {np.mean(model_3_scores)}\n")
    f.write(f"model_3_std_accuracy: {np.std(model_3_scores)}\n")
```

## 5. Model Complexity Analysis

```python
# Analyze model complexity vs. performance
def count_parameters(model):
    return np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])

model_1 = create_model_1(X_train.shape[1], len(np.unique(y)))
model_2 = create_model_2(X_train.shape[1], len(np.unique(y)))
model_3 = create_model_3(X_train.shape[1], len(np.unique(y)))

param_counts = [count_parameters(model_1), count_parameters(model_2), count_parameters(model_3)]

# Plot model complexity vs. performance
plt.figure(figsize=(10, 6))
plt.scatter(param_counts, mean_scores, s=100)

for i, (x, y) in enumerate(zip(param_counts, mean_scores)):
    plt.annotate(model_names[i], (x, y), xytext=(10, 5), textcoords='offset points')

plt.xscale('log')
plt.xlabel('Number of Parameters (log scale)')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Model Complexity vs. Performance')
plt.grid(True, alpha=0.3)
plt.show()

# Save complexity analysis
with open('results/part_1/model_comparison.txt', 'a') as f:
    f.write("\n--- Model Complexity Analysis ---\n")
    for i, name in enumerate(model_names):
        f.write(f"{name}_parameters: {param_counts[i]}\n")
```

## Progress Checkpoints

1. **Data Loading**:
   - [ ] Successfully load healthcare dataset
   - [ ] Verify data shapes and ranges
   - [ ] Split data into train/validation/test sets

2. **Model Definition**:
   - [ ] Define at least 3 different model architectures
   - [ ] Verify architecture differences
   - [ ] Ensure models are properly compiled

3. **Cross-Validation**:
   - [ ] Implement k-fold cross-validation
   - [ ] Evaluate all models using cross-validation
   - [ ] Compare model performance

4. **Best Model Training**:
   - [ ] Train best model with appropriate callbacks
   - [ ] Monitor training progress
   - [ ] Save best model

5. **Evaluation**:
   - [ ] Calculate performance metrics
   - [ ] Analyze model complexity vs. performance
   - [ ] Save metrics in correct format

## Common Issues and Solutions

1. **Data Issues**:
   - Problem: Imbalanced classes
   - Solution: Use class weights or resampling techniques
   - Problem: Feature scaling
   - Solution: Apply standardization or normalization

2. **Model Selection Issues**:
   - Problem: Overfitting in complex models
   - Solution: Add regularization, dropout, or early stopping
   - Problem: Underfitting in simple models
   - Solution: Increase model capacity or feature engineering

3. **Cross-Validation Issues**:
   - Problem: High variance in CV scores
   - Solution: Increase number of folds or use stratified sampling
   - Problem: Slow CV process
   - Solution: Reduce epochs for CV or use a subset of data

4. **Evaluation Issues**:
   - Problem: Metrics format incorrect
   - Solution: Follow the exact format specified
   - Problem: Performance below expectations
   - Solution: Try different architectures or hyperparameter tuning