---
jupyter:
  jupytext:
    notebook_metadata_filter: language_info
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    name: python
    version: 3.13
---

# Demo 3: Flexible Models: Trees, Boosting, and Neural Networks

A breast lump is biopsied with a fine needle, and a digitized image of the cells gives 30 measurements of their nuclei: size, shape, and texture. From those measurements, is the lump malignant? You fit a baseline and logistic regression, then a random forest, gradient-boosted trees with XGBoost, and three small neural networks, all on the same training, validation, and test rows. A selection rule written before any model is fitted picks the winner, and the winner is evaluated on the test rows exactly once. Everything here comes from Lecture 10, plus Lectures 01 to 09. The 569 biopsies are real and de-identified.

**How to run:** open this notebook in Colab from the lecture page's Colab link, or locally in VS Code with the kernel set to a `.venv` made by `uv venv --seed` and `uv pip install -r requirements.txt` in this folder (Lecture 03). Run the cells from top to bottom; after each step, an **Expect** line says what you should see. Colab does not save your changes back to GitHub; use **File → Save a copy in Drive** to keep them. Tested 2026-09-25 with Python 3.13, pandas 3.0.5, NumPy 2.3.3, scikit-learn 1.9.0, XGBoost 2.1.4, TensorFlow 2.21.0, and matplotlib 3.11.1 on a CPU; the whole notebook runs in about a minute. Colab preinstalls other versions of scikit-learn, XGBoost, and TensorFlow. With Colab's XGBoost 3.2, the importances in Part 5 shift in the second decimal place and early stopping in Part 6 can pick a later round; the accuracies do not change. A GPU can change the neural-network numbers.

## Setup

The first cell installs pandas 3.0.5, the course version. Colab ships an older pandas (2.2). A `.venv` made with `uv venv --seed` includes pip, so the same `%pip` cell works locally too.

```python
# Setup: install the course's pandas version (Colab and local)
%pip install -q pandas==3.0.5
```

**Expect:** `Note: you may need to restart the kernel to use updated packages.` Locally, with the requirements already installed, the cell changes nothing and you can go on. In Colab, pip may also print a dependency conflict because some preinstalled packages expect pandas 2.2; that is expected, and this demo does not use them. If Colab asks you to restart the session (or says pandas was previously imported), choose **Runtime → Restart session**, then continue with the next cell. You do not need to rerun the install.

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

# Seed Python, NumPy, and TensorFlow, and make TensorFlow's operations
# deterministic, so the numbers below repeat exactly on a rerun.
keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

print('pandas', pd.__version__)
print('XGBoost', xgb.__version__)
print('TensorFlow', tf.__version__)
```

**Expect:** `pandas 3.0.5`, `XGBoost 2.1.4`, and `TensorFlow 2.21.0` (Colab shows its own XGBoost and TensorFlow versions; that is fine).

## 1. Load the biopsy measurements

`load_breast_cancer(as_frame=True)` ships with scikit-learn. Each row is one biopsy; the 30 columns are ten nucleus measurements (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension), each summarized three ways across the cells in the image: the `mean`, the standard error (`error`), and the `worst` (largest) value. Its target codes malignant as 0, so flip it to make **malignant = 1**, the class we want to catch.

```python
cancer = load_breast_cancer(as_frame=True)
biopsies = cancer.frame
biopsies['malignant'] = (cancer.target == 0).astype(int)
feature_cols = cancer.feature_names.tolist()

print(biopsies.shape)
print(feature_cols[:4])
print(biopsies['malignant'].value_counts())
print(f"Share malignant: {biopsies['malignant'].mean():.1%}")
```

**Expect:** `(569, 32)` (30 measurements, the original `target`, and the new `malignant` column), the first four names `['mean radius', 'mean texture', 'mean perimeter', 'mean area']`, then 357 benign (0) and 212 malignant (1) biopsies: `Share malignant: 37.3%`.

## 2. Split, scale, and fix the selection rule

Reserve the test rows first, then split the rest into training and validation rows. `stratify` keeps the share of malignant biopsies the same in every part.

Trees split on one column at a time, so they use the raw columns. Logistic regression and neural networks train by small repeated adjustments that work far better when every feature is on a similar scale, so they get a scaled copy, with the scaler fitted on the training rows only.

```python
X = biopsies[feature_cols].values
y = biopsies['malignant'].values

X_train_valid, X_test, y_train_valid, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_valid, y_train_valid, test_size=0.25, random_state=42, stratify=y_train_valid)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

print(f"Training:   {X_train.shape}, {y_train.mean():.1%} malignant")
print(f"Validation: {X_valid.shape}, {y_valid.mean():.1%} malignant")
print(f"Test:       {X_test.shape}, {y_test.mean():.1%} malignant")
```

**Expect:**

```text
Training:   (341, 30), 37.2% malignant
Validation: (114, 30), 37.7% malignant
Test:       (114, 30), 36.8% malignant
```

The three parts keep nearly the same share of malignant biopsies, which is what `stratify` is for.

**The selection rule, written down before any model is fitted:**

- The highest validation accuracy wins.
- One validation row is worth 0.9 percentage points of accuracy, so candidates within one row of the best count as tied.
- A tie goes to the _simplest_ candidate: the one with the fewest settings to tune and the fastest fit.

Writing the rule down now stops the final table from being read backwards to justify a favorite. Every model's confusion matrix is printed too, because a missed malignancy (a false negative) is the costliest mistake here.

## 3. A baseline and the simplest candidate

Demo 2 started with `DummyRegressor`; for a yes/no target the same idea is to predict the most common training class for every row. Then fit logistic regression, the linear model for yes/no targets, so every flexible model below has a real bar to clear.

```python
majority_class = int(pd.Series(y_train).mode()[0])
baseline_pred = np.full(len(y_valid), majority_class)
baseline_acc = accuracy_score(y_valid, baseline_pred)
print(f"Majority class in training rows: {majority_class}")
print(f"Baseline validation accuracy: {baseline_acc:.4f}")

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_valid_scaled)
lr_acc = accuracy_score(y_valid, lr_pred)
print(f"Logistic regression validation accuracy: {lr_acc:.4f}")
print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
print(confusion_matrix(y_valid, lr_pred))
print("\n" + classification_report(y_valid, lr_pred))
```

**Expect:**

```text
Majority class in training rows: 0
Baseline validation accuracy: 0.6228
Logistic regression validation accuracy: 0.9737

Confusion matrix [[TN, FP], [FN, TP]]:
[[69  2]
 [ 1 42]]

              precision    recall  f1-score   support

           0       0.99      0.97      0.98        71
           1       0.95      0.98      0.97        43

    accuracy                           0.97       114
   macro avg       0.97      0.97      0.97       114
weighted avg       0.97      0.97      0.97       114
```

Calling every biopsy benign is right 71 times out of 114, so 0.623 is the score to beat, not 0. Logistic regression gets 111 of 114 right: two benign biopsies flagged as malignant, and one malignant biopsy missed. In the report, class `1` (malignant) has recall 0.98: it caught 42 of the 43 malignant biopsies.

## 4. Random forest, and two ways to read it

A random forest grows many trees on random resamples of the training rows and averages their votes. It can capture curved relationships and interactions, and it needs no scaling.

```python
rf_model = RandomForestClassifier(
    n_estimators=100,     # number of trees
    max_depth=5,          # maximum depth of each tree
    min_samples_split=5,  # rows a question needs before it splits
    random_state=42,
    n_jobs=-1,
)
rf_model.fit(X_train, y_train)

rf_acc = accuracy_score(y_valid, rf_model.predict(X_valid))
print(f"Training accuracy:   {accuracy_score(y_train, rf_model.predict(X_train)):.4f}")
print(f"Validation accuracy: {rf_acc:.4f}")
print("Confusion matrix [[TN, FP], [FN, TP]]:")
print(confusion_matrix(y_valid, rf_model.predict(X_valid)))
print(f"First three predicted probabilities (benign, malignant):\n{rf_model.predict_proba(X_valid)[:3].round(3)}")
```

**Expect:**

```text
Training accuracy:   0.9941
Validation accuracy: 0.9737
Confusion matrix [[TN, FP], [FN, TP]]:
[[70  1]
 [ 2 41]]
First three predicted probabilities (benign, malignant):
[[0.997 0.003]
 [0.066 0.934]
 [0.998 0.002]]
```

Nearly perfect on the rows it learned and 111 of 114 on rows it has never seen, the same count as logistic regression, but with a different mix of mistakes: one false alarm and two missed malignancies instead of two and one.

Which columns does the forest lean on? There are two answers, measured on different data. `feature_importances_` comes from the _training_ fit and exists only for tree models. `permutation_importance` shuffles one validation column at a time and measures how far accuracy falls, so it reports held-out reliance for any fitted model.

```python
perm = permutation_importance(
    rf_model, X_valid, y_valid,
    scoring='accuracy', n_repeats=10, random_state=42, n_jobs=-1,
)
rf_importance = pd.DataFrame({
    'feature': feature_cols,
    'impurity': rf_model.feature_importances_,
    'accuracy_drop': perm.importances_mean,
    'drop_std': perm.importances_std,
}).sort_values('impurity', ascending=False)
print(rf_importance.head(8).round(3).to_string(index=False))
```

**Expect:**

```text
             feature  impurity  accuracy_drop  drop_std
worst concave points     0.140          0.013     0.010
          worst area     0.135          0.014     0.011
 mean concave points     0.112          0.000     0.000
     worst perimeter     0.098          0.011     0.007
         mean radius     0.068          0.000     0.000
      mean perimeter     0.068          0.000     0.000
      mean concavity     0.068          0.005     0.004
        worst radius     0.062          0.006     0.006
```

The two columns agree at the top: `worst concave points` and `worst area` matter by both measures. Lower down they disagree. `mean concave points` is third by impurity, yet shuffling it costs no validation accuracy at all, and neither does shuffling `mean radius` or `mean perimeter`. Many of these 30 columns measure nearly the same thing (radius, perimeter, and area all describe size), so when one is shuffled the forest still has its twins, and the importance spreads across the group. Read the top of this ranking, not the bottom, and read neither column as a causal effect.

## 5. XGBoost

XGBoost builds its trees in sequence, each one aimed at what the ensemble so far still gets wrong. It follows the same fit/predict pattern.

```python
xgb_model = xgb.XGBClassifier(
    n_estimators=100, max_depth=3, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1,
)
xgb_model.fit(X_train, y_train)
xgb_acc = accuracy_score(y_valid, xgb_model.predict(X_valid))
print(f"XGBoost validation accuracy: {xgb_acc:.4f}")
print("Confusion matrix [[TN, FP], [FN, TP]]:")
print(confusion_matrix(y_valid, xgb_model.predict(X_valid)))

top_features = rf_importance['feature'].head(8).tolist()
importance_comparison = pd.DataFrame({
    'random forest': rf_model.feature_importances_,
    'XGBoost': xgb_model.feature_importances_,
}, index=feature_cols).loc[top_features]
print(importance_comparison.round(3))

fig, ax = plt.subplots(figsize=(9, 4))
importance_comparison.plot(kind='bar', ax=ax)
ax.set_ylabel('Impurity-based importance')
ax.set_title("The random forest's top eight features, in both models")
ax.tick_params(axis='x', rotation=45)
plt.show()
plt.close(fig)
```

**Expect:**

```text
XGBoost validation accuracy: 0.9649
Confusion matrix [[TN, FP], [FN, TP]]:
[[69  2]
 [ 2 41]]
                      random forest  XGBoost
worst concave points          0.140    0.084
worst area                    0.135    0.038
mean concave points           0.112    0.167
worst perimeter               0.098    0.251
mean radius                   0.068    0.004
mean perimeter                0.068    0.000
mean concavity                0.068    0.058
worst radius                  0.062    0.152
```

XGBoost gets 110 of 114, one fewer than the forest. In the bar chart it leans hardest on `worst perimeter` (0.251), which the forest ranks fourth, and it ignores `mean perimeter` entirely. Each library computes importance its own way, and among near-twin columns a model can pick any one, so compare rankings, not numbers.

## 6. Early stopping

Early stopping scores the validation rows after every boosting round and stops when the score stops improving, so you can set a generous round budget. Because the validation rows now help _fit_ this model, its validation score is no longer on equal footing with the others.

```python
xgb_early_stop = xgb.XGBClassifier(
    n_estimators=500,            # a generous budget; early stopping decides where to stop
    max_depth=3, learning_rate=0.1,
    early_stopping_rounds=10,    # in the constructor since XGBoost 2.0
    random_state=42, n_jobs=-1,
)
xgb_early_stop.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
xgb_early_acc = accuracy_score(y_valid, xgb_early_stop.predict(X_valid))

print(f"Best iteration: {xgb_early_stop.best_iteration} (of a 500-round budget)")
print(f"Best validation loss: {xgb_early_stop.best_score:.4f}")
print(f"XGBoost, 100 fixed rounds: validation accuracy {xgb_acc:.4f}")
print(f"XGBoost, early stopped:    validation accuracy {xgb_early_acc:.4f}")
```

**Expect:**

```text
Best iteration: 420 (of a 500-round budget)
Best validation loss: 0.0921
XGBoost, 100 fixed rounds: validation accuracy 0.9649
XGBoost, early stopped:    validation accuracy 0.9825
```

On a table this small the validation loss keeps creeping down for hundreds of rounds, so early stopping uses most of the budget. The early-stopped model gets 112 of 114, the best score so far, and those same 114 rows chose its round count. Part 10 comes back to that.

## 7. Build and compile a neural network

A `Sequential` model is a straight stack of layers. Two hidden ReLU layers followed by one sigmoid unit is a standard starting point for a yes/no target. Compiling attaches the **optimizer** that updates the weights (Adam), the **loss** it minimizes (`binary_crossentropy` for a yes/no target), and the metrics reported each epoch.

```python
n_features = X_train.shape[1]  # 30 nucleus measurements

model = keras.Sequential([
    keras.layers.Input(shape=(n_features,)),
    keras.layers.Dense(64, activation='relu', name='hidden1'),
    keras.layers.Dense(32, activation='relu', name='hidden2'),
    keras.layers.Dense(1, activation='sigmoid', name='output'),
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
```

**Expect:** a table with three `Dense` layers of 1,984, 2,080, and 33 weights, and `Total params: 4,097 (16.00 KB)`. The first layer has 30 × 64 weights plus 64 intercepts: 1,984. That is 4,097 weights for 341 training rows, which is why the loss curves in the next part are worth watching.

## 8. Train and read the curves

An **epoch** is one pass through the training rows; `batch_size=32` updates the weights after every 32 rows. `validation_data` scores the validation rows after every epoch, which is what lets you diagnose the training.

```python
history = model.fit(
    X_train_scaled, y_train,
    epochs=50, batch_size=32,
    validation_data=(X_valid_scaled, y_valid),
    verbose=0,
)

history_df = pd.DataFrame(history.history)
history_df['epoch'] = range(1, len(history_df) + 1)
print(history_df.tail(3).round(4).to_string(index=False))

best_row = history_df.loc[history_df['val_loss'].idxmin()]
print(f"\nLowest validation loss: {best_row['val_loss']:.4f} at epoch {best_row['epoch']:.0f}")

valid_loss, valid_accuracy = model.evaluate(X_valid_scaled, y_valid, verbose=0)
print(f"Validation loss after epoch 50: {valid_loss:.4f}")
print(f"Validation accuracy: {valid_accuracy:.4f}")
nn_pred = (model.predict(X_valid_scaled, verbose=0) > 0.5).astype(int).flatten()
print("Confusion matrix [[TN, FP], [FN, TP]]:")
print(confusion_matrix(y_valid, nn_pred))
```

**Expect:** TensorFlow may first print log lines in red, such as `WARNING: All log messages before absl::InitializeLog() is called are written to STDERR` or a line starting `E0000` that mentions `use_unbounded_threadpool`. They come from TensorFlow's internals, not from your code, and do not change the results. Then:

```text
 accuracy   loss  val_accuracy  val_loss  epoch
      1.0 0.0075        0.9737    0.0896     48
      1.0 0.0071        0.9737    0.0898     49
      1.0 0.0067        0.9737    0.0903     50

Lowest validation loss: 0.0726 at epoch 20
Validation loss after epoch 50: 0.0903
Validation accuracy: 0.9737
Confusion matrix [[TN, FP], [FN, TP]]:
[[69  2]
 [ 1 42]]
```

The network gets every training row right and 111 of 114 validation rows, with the same confusion matrix as logistic regression.

```python
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(history_df['epoch'], history_df['loss'], label='training loss')
ax.plot(history_df['epoch'], history_df['val_loss'], label='validation loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Binary cross-entropy loss')
ax.set_title('Training and validation loss')
ax.legend()
plt.show()
plt.close(fig)
```

**Expect:** both curves fall steeply for the first ten epochs. After that, training loss keeps falling toward 0, while validation loss bottoms out at epoch 20 (0.0726) and then creeps back up to 0.0903. That widening gap is overfitting starting: the network is memorizing its 341 training rows, and it is the signal early stopping watches for. Accuracy has not suffered yet, but the model is growing more confident than the validation rows justify.

## 9. Two variants

Depth, width, and regularization are settings to validate, not upgrades. Train one variant with more layers and units, and one with **dropout** (randomly masking 30% of a layer's outputs during training) plus an L2 penalty on the weights. Same epochs and batch size, so the comparison is fair.

Compare the three on validation accuracy, not loss. Keras adds the L2 penalty to the loss it reports, on training and validation rows alike, so the regularized network's loss is on a different scale from the other two.

```python
model_deep = keras.Sequential([
    keras.layers.Input(shape=(n_features,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
])
model_deep.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_deep.fit(X_train_scaled, y_train, epochs=50, batch_size=32,
               validation_data=(X_valid_scaled, y_valid), verbose=0)
deep_loss, deep_acc = model_deep.evaluate(X_valid_scaled, y_valid, verbose=0)
deep_pred = (model_deep.predict(X_valid_scaled, verbose=0) > 0.5).astype(int).flatten()

model_reg = keras.Sequential([
    keras.layers.Input(shape=(n_features,)),
    keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1, activation='sigmoid'),
])
model_reg.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_reg = model_reg.fit(X_train_scaled, y_train, epochs=50, batch_size=32,
                            validation_data=(X_valid_scaled, y_valid), verbose=0)
reg_loss, reg_acc = model_reg.evaluate(X_valid_scaled, y_valid, verbose=0)
reg_pred = (model_reg.predict(X_valid_scaled, verbose=0) > 0.5).astype(int).flatten()

print(pd.DataFrame({
    'network': ['64-32', '128-64-32-16', '64-32 + dropout/L2'],
    'weights': [model.count_params(), model_deep.count_params(), model_reg.count_params()],
    'valid_accuracy': [valid_accuracy, deep_acc, reg_acc],
}).round(4).to_string(index=False))

reg_history_df = pd.DataFrame(history_reg.history)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(history_df['epoch'], history_df['loss'], label='training loss')
axes[0].plot(history_df['epoch'], history_df['val_loss'], label='validation loss')
axes[0].set_title('Plain 64-32')
axes[1].plot(history_df['epoch'], reg_history_df['loss'], label='training loss')
axes[1].plot(history_df['epoch'], reg_history_df['val_loss'], label='validation loss')
axes[1].set_title('64-32 + dropout/L2 (loss includes the L2 penalty)')
for ax in axes:
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
plt.show()
plt.close(fig)
```

**Expect:** TensorFlow may print a `WARNING:tensorflow:... triggered tf.function retracing` message while the networks predict one after another; it is about speed, not a failure. Then:

```text
           network  weights  valid_accuracy
             64-32     4097          0.9737
      128-64-32-16    14849          0.9649
64-32 + dropout/L2     4097          0.9737
```

More than three times the weights buys one _fewer_ correct validation row (110 of 114). In the plots, the plain network's validation loss turns up after epoch 20, while the regularized network's training and validation curves fall together all the way to epoch 50: dropout and the L2 penalty closed the gap. It did not change the accuracy, 111 of 114. On a table this small, regularization is a setting to validate, not an upgrade.

## 10. Compare every candidate

One validation set, one metric, and every model this notebook fitted, including the early-stopped XGBoost, whose score had help from the validation rows.

```python
comparison = pd.DataFrame({
    'model': ['Majority-class baseline', 'Logistic regression', 'Random forest',
              'XGBoost (100 rounds)', 'XGBoost (early stopped)', 'Neural net 64-32',
              'Neural net 128-64-32-16', 'Neural net 64-32 + dropout/L2'],
    'valid_accuracy': [baseline_acc, lr_acc, rf_acc, xgb_acc, xgb_early_acc,
                       valid_accuracy, deep_acc, reg_acc],
})
comparison['correct_of_114'] = (comparison['valid_accuracy'] * len(y_valid)).round().astype(int)
print(comparison.round(4).to_string(index=False))
```

**Expect:**

```text
                        model  valid_accuracy  correct_of_114
      Majority-class baseline          0.6228              71
          Logistic regression          0.9737             111
                Random forest          0.9737             111
         XGBoost (100 rounds)          0.9649             110
      XGBoost (early stopped)          0.9825             112
             Neural net 64-32          0.9737             111
      Neural net 128-64-32-16          0.9649             110
Neural net 64-32 + dropout/L2          0.9737             111
```

Every model clears the baseline by a wide margin, and all seven land within two rows of each other.

Identical scores do not mean identical predictions. List the validation rows each model gets wrong, and which of those are missed malignancies.

```python
valid_predictions = {
    'Logistic regression': lr_pred,
    'Random forest': rf_model.predict(X_valid),
    'XGBoost (100 rounds)': xgb_model.predict(X_valid),
    'XGBoost (early stopped)': xgb_early_stop.predict(X_valid),
    'Neural net 64-32': nn_pred,
    'Neural net 128-64-32-16': deep_pred,
    'Neural net 64-32 + dropout/L2': reg_pred,
}

row_numbers = np.arange(len(y_valid))
for name, pred in valid_predictions.items():
    wrong = row_numbers[pred != y_valid]
    missed_malignant = row_numbers[(pred == 0) & (y_valid == 1)]
    print(f"{name:30s} wrong on rows {wrong.tolist()}, missed malignant {missed_malignant.tolist()}")
```

**Expect:**

```text
Logistic regression            wrong on rows [9, 10, 75], missed malignant [9]
Random forest                  wrong on rows [9, 10, 45], missed malignant [9, 45]
XGBoost (100 rounds)           wrong on rows [9, 10, 45, 98], missed malignant [9, 45]
XGBoost (early stopped)        wrong on rows [9, 98], missed malignant [9]
Neural net 64-32               wrong on rows [9, 10, 75], missed malignant [9]
Neural net 128-64-32-16        wrong on rows [9, 10, 34, 98], missed malignant [9]
Neural net 64-32 + dropout/L2  wrong on rows [9, 10, 75], missed malignant [9]
```

- **Row 9** is a malignant biopsy that every model calls benign. No choice of model family fixes it; it is worth a pathologist's look.
- **Same score, different mistakes:** logistic regression and the forest both get 111 right, but the forest misses a second malignancy (row 45) where logistic regression raises a false alarm (row 75). For a screening tool that difference matters more than the tie in accuracy.
- **The early-stopped XGBoost** is right on rows 10, 45, and 75, and its 112 of 114 is the best in the table. Those same 114 rows chose its round count, so its score is the most optimistic number here, not the most trustworthy.

Now apply Part 2's rule. The best candidate (112) and the runners-up (111) differ by one validation row, so they are tied, and the tie goes to the simplest candidate: **logistic regression**, 30 coefficients and an intercept, fitted in milliseconds. That is the honest reading of 114 validation rows. Deep learning did not fail; this table is too small to tell these families apart.

## 11. Freeze, then test once

Freezing means the features, preprocessing, and settings stop changing. Refit the frozen configuration on the training and validation rows together (more data, no new choices), with a scaler fitted on those same rows, and score it on the test rows exactly once.

```python
final_scaler = StandardScaler()
X_train_valid_scaled = final_scaler.fit_transform(X_train_valid)
X_test_scaled = final_scaler.transform(X_test)

final_model = LogisticRegression(max_iter=1000, random_state=42)
final_model.fit(X_train_valid_scaled, y_train_valid)

test_pred = final_model.predict(X_test_scaled)
print("Final test performance, frozen logistic regression:")
print(f"Test accuracy: {accuracy_score(y_test, test_pred):.4f}")
print("Confusion matrix [[TN, FP], [FN, TP]]:")
print(confusion_matrix(y_test, test_pred))
```

**Expect:**

```text
Final test performance, frozen logistic regression:
Test accuracy: 0.9649
Confusion matrix [[TN, FP], [FN, TP]]:
[[71  1]
 [ 3 39]]
```

110 of 114 test biopsies are right: one false alarm and three missed malignancies out of 42. Test accuracy (0.965) is a little below validation (0.974), the normal cost of choosing on validation rows. Three missed cancers is the number to take to the clinical team. If misses cost that much more than false alarms, the next project would pick its threshold or metric for recall before looking at the test set again; going back now to try the early-stopped XGBoost on these rows would turn the test set into a second validation set.
