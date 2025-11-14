# Demo 3: Deep Learning with TensorFlow/Keras

## Learning Objectives
- Build neural networks using TensorFlow/Keras
- Understand the Sequential API
- Train models and monitor progress
- Evaluate model performance
- Visualize training history
- Compare deep learning with traditional ML

## Setup

**Important:** This demo requires Python 3.13 or earlier. When creating your virtual environment with `uv`, use: `uv venv --python python3.13`

This ensures TensorFlow can be installed. If you're using Python 3.14, TensorFlow is not yet available.

```python
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import altair as alt
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")
```

## Part 1: Load Real Classification Dataset

For deep learning, we'll use the Wine Quality dataset - a real-world dataset containing chemical properties of wines and their quality ratings. We'll convert this to a binary classification problem.

```python
# Load Wine Quality dataset from scikit-learn
from sklearn.datasets import load_wine

# Fetch the dataset
wine_data = load_wine(as_frame=True)
df = wine_data.frame

# The dataset contains 13 features describing wine chemical properties:
# - Alcohol, Malic acid, Ash, Alkalinity of ash, Magnesium
# - Total phenols, Flavanoids, Nonflavanoid phenols, Proanthocyanins
# - Color intensity, Hue, OD280/OD315, Proline
# - Target: wine class (0, 1, or 2 - three types of wine)

# Convert to binary classification: class 0 vs others
df['target'] = (wine_data.target == 0).astype(int)

print("Dataset shape:", df.shape)
print("\nFeature names:", wine_data.feature_names)
print("\nFirst few rows:")
print(df.head())
print(f"\nTarget distribution:")
print(df['target'].value_counts())
print(f"\nClass balance: {df['target'].mean():.2%} positive class (wine type 0)")
print("\nSummary statistics:")
print(df.describe())
```

## Part 2: Data Preprocessing

Neural networks work best with scaled features. Let's prepare our data.

Neural networks are sensitive to the scale of input features. Unlike tree-based models (Random Forest, XGBoost) which can handle different scales, neural networks use gradient descent optimization that works much better when all features are on a similar scale.

**Why scaling matters:**
- Features with larger values can dominate the learning process
- Gradient descent converges faster with scaled features
- Activation functions work better when inputs are in a reasonable range
- Without scaling, some features might be ignored or cause training instability

```python
# Split into features and target
# Use all wine chemical properties as features
feature_cols = wine_data.feature_names
X = df[feature_cols].values
y = df['target'].values

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (important for neural networks!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train_scaled.shape}")
print(f"Test set: {X_test_scaled.shape}")
print(f"\nFeature statistics (after scaling):")
print(f"Mean: {X_train_scaled.mean(axis=0)[:5]}")  # Should be ~0
print(f"Std: {X_train_scaled.std(axis=0)[:5]}")    # Should be ~1
```

**StandardScaler** transforms features to have mean=0 and standard deviation=1. Notice we fit the scaler on training data only, then transform both training and test data. This prevents data leakage - the test set statistics shouldn't influence the scaling.

## Part 3: Build Your First Neural Network

Let's create a simple neural network using Keras Sequential API.

```python
# Build a simple neural network
n_features = X_train.shape[1]  # Number of input features (13 for wine dataset)

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(n_features,), name='hidden1'),
    keras.layers.Dense(32, activation='relu', name='hidden2'),
    keras.layers.Dense(1, activation='sigmoid', name='output')  # Binary classification
])

# Display model architecture
print("=== Model Architecture ===")
model.summary()

# Visualize model (optional, requires graphviz)
# keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)
```

**Understanding the architecture:**
- **Input layer**: 20 features (automatically created)
- **Hidden layer 1**: 64 neurons with ReLU activation
- **Hidden layer 2**: 32 neurons with ReLU activation
- **Output layer**: 1 neuron with sigmoid activation (for binary classification)

## Part 4: Compile the Model

Before training, we need to specify the optimizer, loss function, and metrics.

Before training, we need to configure three key components:

1. **Optimizer**: How the model updates its weights during training (Adam is a popular choice)
2. **Loss function**: What the model tries to minimize (binary_crossentropy for classification)
3. **Metrics**: What we track during training (accuracy tells us how often predictions are correct)

```python
# Compile the model
model.compile(
    optimizer='adam',  # Adaptive learning rate optimizer
    loss='binary_crossentropy',  # For binary classification
    metrics=['accuracy']  # Track accuracy during training
)

print("Model compiled successfully!")
print(f"Optimizer: {model.optimizer.get_config()['name']}")
print(f"Loss function: {model.loss}")
print(f"Metrics: {[m.name for m in model.metrics]}")
```

**Understanding these choices:**
- **Adam optimizer**: Adapts the learning rate for each parameter, making training more efficient
- **Binary crossentropy**: Appropriate for binary classification (two classes)
- **Accuracy**: Simple metric - percentage of correct predictions. For imbalanced classes, you might also track precision/recall.

## Part 5: Train the Model

Now let's train the model and watch it learn!

```python
# Train the model
history = model.fit(
    X_train_scaled, y_train,
    epochs=50,  # Number of training iterations
    batch_size=32,  # Number of samples per gradient update
    validation_split=0.2,  # Use 20% of training data for validation
    verbose=1  # Show progress
)
```

**Understanding training:**
- **Epoch**: One pass through the entire training dataset
- **Batch size**: Number of samples processed before updating weights
- **Validation split**: Hold out some training data to monitor overfitting

## Part 6: Evaluate Model Performance

Let's see how well our model performs on the test set.

```python
# Evaluate on test set
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"=== Test Set Performance ===")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Make predictions
y_pred_proba = model.predict(X_test_scaled, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Classification report
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\n=== Confusion Matrix ===")
print("                Predicted")
print("              Negative  Positive")
print(f"Actual Negative    {cm[0,0]:4d}     {cm[0,1]:4d}")
print(f"        Positive    {cm[1,0]:4d}     {cm[1,1]:4d}")
```

## Part 7: Visualize Training History

Let's plot how the model learned over time.

```python
# Extract training history
history_df = pd.DataFrame(history.history)
history_df['epoch'] = range(1, len(history_df) + 1)

print("=== Training History ===")
print(history_df.tail())

# Plot training curves
history_long = history_df.melt(
    id_vars='epoch',
    value_vars=['loss', 'val_loss', 'accuracy', 'val_accuracy'],
    var_name='metric',
    value_name='value'
)

# Separate loss and accuracy
loss_data = history_long[history_long['metric'].isin(['loss', 'val_loss'])]
acc_data = history_long[history_long['metric'].isin(['accuracy', 'val_accuracy'])]

# Loss plot
loss_chart = alt.Chart(loss_data).mark_line(point=True).encode(
    x=alt.X('epoch:Q', title='Epoch'),
    y=alt.Y('value:Q', title='Loss'),
    color='metric:N',
    strokeDash=alt.condition(alt.datum.metric == 'val_loss', alt.value([5, 5]), alt.value([0]))
).properties(
    width=400,
    height=250,
    title='Training and Validation Loss'
)

# Accuracy plot
acc_chart = alt.Chart(acc_data).mark_line(point=True).encode(
    x=alt.X('epoch:Q', title='Epoch'),
    y=alt.Y('value:Q', title='Accuracy', scale=alt.Scale(domain=[0, 1])),
    color='metric:N',
    strokeDash=alt.condition(alt.datum.metric == 'val_accuracy', alt.value([5, 5]), alt.value([0]))
).properties(
    width=400,
    height=250,
    title='Training and Validation Accuracy'
)

# Combine charts
alt.vconcat(loss_chart, acc_chart)
```

**What to look for:**
- **Loss decreasing**: Model is learning
- **Validation loss tracking training loss**: No overfitting
- **Gap between train/val**: If validation loss increases while training decreases, you're overfitting

## Part 8: Compare with Traditional ML

Let's see how deep learning compares to traditional ML methods on this dataset.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, lr_pred)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
rf_pred = rf.predict(X_test_scaled)
rf_acc = accuracy_score(y_test, rf_pred)

# XGBoost
xgb_clf = xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1)
xgb_clf.fit(X_train_scaled, y_train)
xgb_pred = xgb_clf.predict(X_test_scaled)
xgb_acc = accuracy_score(y_test, xgb_pred)

# Compare
comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost', 'Neural Network'],
    'Accuracy': [lr_acc, rf_acc, xgb_acc, test_accuracy]
})

print("=== Model Comparison ===")
print(comparison.to_string(index=False))

# Visualize
alt.Chart(comparison).mark_bar().encode(
    x=alt.X('Model:N', title='Model', sort='-y'),
    y=alt.Y('Accuracy:Q', title='Test Accuracy', scale=alt.Scale(domain=[0, 1]))
).properties(
    width=400,
    height=300
)
```

**Key insight**: On tabular data, traditional ML (especially XGBoost) often performs as well or better than deep learning, with less complexity and faster training!

## Part 9: Experiment with Architecture

Let's try different architectures to see how they affect performance.

```python
# Build a deeper network
model_deep = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(n_features,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model_deep.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train deeper model
history_deep = model_deep.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

# Evaluate
deep_test_loss, deep_test_acc = model_deep.evaluate(X_test_scaled, y_test, verbose=0)

# Build a wider network
model_wide = keras.Sequential([
    keras.layers.Dense(256, activation='relu', input_shape=(n_features,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model_wide.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train wider model
history_wide = model_wide.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

# Evaluate
wide_test_loss, wide_test_acc = model_wide.evaluate(X_test_scaled, y_test, verbose=0)

# Compare architectures
arch_comparison = pd.DataFrame({
    'Architecture': ['Original (64-32)', 'Deep (128-64-32-16)', 'Wide (256-128)'],
    'Test Accuracy': [test_accuracy, deep_test_acc, wide_test_acc],
    'Parameters': [model.count_params(), model_deep.count_params(), model_wide.count_params()]
})

print("=== Architecture Comparison ===")
print(arch_comparison.to_string(index=False))
```

**Insights:**
- More layers (depth) doesn't always mean better performance
- More neurons (width) increases model capacity but also risk of overfitting
- Find the right balance for your specific problem

## Part 10: Regularization Techniques

Let's add dropout and L2 regularization to prevent overfitting.

```python
# Model with regularization
model_regularized = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(n_features,),
                       kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dropout(0.3),  # Drop 30% of neurons randomly
    keras.layers.Dense(32, activation='relu',
                       kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1, activation='sigmoid')
])

model_regularized.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train with regularization
history_reg = model_regularized.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

# Evaluate
reg_test_loss, reg_test_acc = model_regularized.evaluate(X_test_scaled, y_test, verbose=0)

print("=== Regularization Comparison ===")
print(f"Original model - Test Accuracy: {test_accuracy:.4f}")
print(f"Regularized model - Test Accuracy: {reg_test_acc:.4f}")

# Compare training curves
history_reg_df = pd.DataFrame(history_reg.history)
history_reg_df['epoch'] = range(1, len(history_reg_df) + 1)

# Plot validation loss comparison
val_loss_comparison = pd.DataFrame({
    'epoch': history_df['epoch'],
    'original': history_df['val_loss'],
    'regularized': history_reg_df['val_loss']
}).melt(
    id_vars='epoch',
    value_vars=['original', 'regularized'],
    var_name='model',
    value_name='val_loss'
)

alt.Chart(val_loss_comparison).mark_line(point=True).encode(
    x='epoch:Q',
    y='val_loss:Q',
    color='model:N'
).properties(
    width=400,
    height=250,
    title='Validation Loss: Original vs Regularized'
)
```

**Regularization techniques:**
- **L2 regularization**: Penalizes large weights
- **Dropout**: Randomly disables neurons during training (prevents co-adaptation)
- Both help prevent overfitting

## Key Takeaways

1. **Sequential API**: Simple way to build linear stacks of layers
2. **Data scaling**: Always scale features for neural networks
3. **Compile step**: Specify optimizer, loss, and metrics
4. **Training**: Monitor both training and validation metrics
5. **Architecture matters**: Experiment with depth and width
6. **Regularization**: Use dropout and L2 to prevent overfitting
7. **Deep learning isn't always better**: For tabular data, traditional ML often wins
8. **Use deep learning when**: You have images, text, sequences, or massive datasets

## When to Use Deep Learning

- ✅ **Images**: Computer vision (CNNs)
- ✅ **Text**: Natural language processing (RNNs, Transformers)
- ✅ **Sequences**: Time series, audio (RNNs, LSTMs)
- ✅ **Massive datasets**: Millions of examples
- ❌ **Tabular data**: Often better with XGBoost
- ❌ **Small datasets**: Deep learning needs lots of data
- ❌ **Need interpretability**: Neural networks are black boxes

## Next Steps

- Explore different activation functions (tanh, LeakyReLU)
- Try different optimizers (RMSprop, SGD with momentum)
- Learn about callbacks (EarlyStopping, ModelCheckpoint)
- Experiment with different architectures
- Explore PyTorch for more flexibility

