# Demo 3: Flexible Models - Trees, Boosting, and Neural Networks

## Learning Objectives
- Fit a random forest and read its importances two ways: impurity and permutation
- Fit gradient-boosted trees with XGBoost, including early stopping
- Build, compile, and train a neural network with TensorFlow/Keras
- Read training and validation loss curves
- Compare every candidate on one validation set and apply a selection rule fixed in advance
- Freeze the winner and evaluate it on the test set exactly once

## Setup

Use the course Python 3.13 runtime with TensorFlow 2.21.0 and the exact package versions in `requirements.txt`.

```python
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import altair as alt

# Seed everything, and make TensorFlow's operations deterministic, so the
# numbers printed below repeat exactly on a rerun.
keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

print(f"TensorFlow version: {tf.__version__}")
print(f"XGBoost version: {xgb.__version__}")
```

## Part 1: Load a Classification Dataset

Every model in this demo predicts the same yes/no target, so they can be compared
fairly. We use scikit-learn's Wine recognition dataset: chemical measurements for
three cultivars, collapsed to a binary problem (class 0 versus the rest).

```python
from sklearn.datasets import load_wine

wine_data = load_wine(as_frame=True)
df = wine_data.frame

# Binary target: is this wine cultivar 0?
df['target'] = (wine_data.target == 0).astype(int)
feature_cols = wine_data.feature_names

print("Dataset shape:", df.shape)
print("\nFeatures:", feature_cols)
print(f"\nTarget distribution:\n{df['target'].value_counts()}")
print(f"\nClass balance: {df['target'].mean():.2%} positive")
```

Only 178 rows, so one validation row is worth 2.8 percentage points of accuracy.
Keep that in mind before declaring a winner.

## Part 2: Split, Scale, and Fix the Selection Rule

Reserve the test set first, then split the remainder into training and validation
rows. `stratify` keeps the class mix the same in every part, which matters when a
part holds only 36 rows.

Tree models split on one column at a time, so they work on the raw columns.
Logistic regression and neural networks use gradient descent, which converges far
better when every feature is on a similar scale, so they get a scaled copy. The
scaler is fitted on the training rows only; validation and test statistics never
touch it.

```python
X = df[feature_cols].values
y = df['target'].values

X_train_valid, X_test, y_train_valid, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_valid, y_train_valid, test_size=0.25, random_state=42, stratify=y_train_valid
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_valid.shape}")
print(f"Test set: {X_test.shape}")
print(f"Scaled training means (first 3): {X_train_scaled.mean(axis=0)[:3].round(3)}")
print(f"Scaled training stds  (first 3): {X_train_scaled.std(axis=0)[:3].round(3)}")
```

```text
Training set: (106, 13)
Validation set: (36, 13)
Test set: (36, 13)
Scaled training means (first 3): [0. 0. 0.]
Scaled training stds  (first 3): [1. 1. 1.]
```

**The selection rule, written down before any model is fitted:** pick the
candidate with the highest validation accuracy; if the best and the runner-up are
within one validation row of each other, that gap is noise on 36 rows, so treat
them as tied and keep the *simplest* candidate - the one with the fewest settings
to tune and the fastest fit. Writing the rule down now is what stops the table at
the end from being read backwards to justify a favourite.

## Part 3: A Baseline and the Simplest Candidate

Demo 2 started with `DummyRegressor`; the classification version of the same idea
is to predict the most common training class for every row. Then fit logistic
regression - Demo 2's linear pipeline with a yes/no target - so every flexible
model below has a real bar to clear.

```python
majority_class = int(pd.Series(y_train).mode()[0])
baseline_pred = np.full(shape=y_valid.shape, fill_value=majority_class)
baseline_acc = accuracy_score(y_valid, baseline_pred)
print(f"Majority class in training rows: {majority_class}")
print(f"Baseline validation accuracy: {baseline_acc:.4f}")

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
lr_acc = accuracy_score(y_valid, lr.predict(X_valid_scaled))

print(f"Logistic regression validation accuracy: {lr_acc:.4f}")
print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
print(confusion_matrix(y_valid, lr.predict(X_valid_scaled)))
print("\n" + classification_report(y_valid, lr.predict(X_valid_scaled)))
```

```text
Majority class in training rows: 0
Baseline validation accuracy: 0.6667
Logistic regression validation accuracy: 0.9722

Confusion matrix [[TN, FP], [FN, TP]]:
[[24  0]
 [ 1 11]]
```

Always answering "not cultivar 0" is right 24 times out of 36, so 0.667 is the
score to beat, not 0. Logistic regression makes one false negative out of 36: it
missed a single cultivar-0 wine and got everything else right. That is the number
every model below has to beat.

## Part 4: Random Forest, and Two Ways to Read It

A random forest grows many trees on random resamples of the training rows and
averages their votes. It captures non-linear relationships and interactions
automatically, and it needs no scaling.

```python
rf_model = RandomForestClassifier(
    n_estimators=100,     # Number of trees
    max_depth=5,          # Maximum tree depth
    min_samples_split=5,  # Minimum samples to split
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

rf_train_acc = accuracy_score(y_train, rf_model.predict(X_train))
rf_acc = accuracy_score(y_valid, rf_model.predict(X_valid))
print(f"Training accuracy:   {rf_train_acc:.4f}")
print(f"Validation accuracy: {rf_acc:.4f}")
print(f"First three predicted probabilities:\n{rf_model.predict_proba(X_valid)[:3].round(3)}")
```

```text
Training accuracy:   1.0000
Validation accuracy: 0.9722
First three predicted probabilities:
[[0.92  0.08 ]
 [0.764 0.236]
 [0.994 0.006]]
```

Perfect on the rows it learned, 35 of 36 on rows it has never seen: the forest
memorised 106 rows and still generalises.

Now the question the accuracy cannot answer: which columns is it leaning on?
There are two answers, and they are measured on different data.
`feature_importances_` comes from the *training* fit and is defined only for tree
models. `permutation_importance` shuffles one validation column at a time and
measures how far accuracy falls, so it works on any fitted estimator and reports
held-out reliance.

```python
perm = permutation_importance(
    rf_model, X_valid, y_valid,
    scoring='accuracy', n_repeats=10, random_state=42, n_jobs=-1,
)

rf_importance = pd.DataFrame({
    'feature': feature_cols,
    'impurity_importance': rf_model.feature_importances_.round(3),
    'validation_accuracy_drop': perm.importances_mean.round(3),
    'repeat_std': perm.importances_std.round(3),
}).sort_values('impurity_importance', ascending=False)

print(rf_importance.head(8).to_string(index=False))
```

```text
                     feature  impurity_importance  validation_accuracy_drop  repeat_std
                     proline                0.251                     0.081       0.038
                  flavanoids                0.214                     0.033       0.035
                     alcohol                0.136                     0.019       0.025
               total_phenols                0.101                     0.025       0.019
           alcalinity_of_ash                0.060                    -0.017       0.014
             color_intensity                0.055                    -0.011       0.018
                   magnesium                0.044                     0.022       0.011
od280/od315_of_diluted_wines                0.038                     0.033       0.011
```

The two columns agree on `proline` and disagree lower down.
`alcalinity_of_ash` and `color_intensity` sit fifth and sixth by impurity, yet
shuffling either one leaves validation accuracy *slightly better* (-0.017 and
-0.011): the trees used them to carve up the training rows, but they buy nothing
on held-out rows. `od280/od315_of_diluted_wines` is the mirror image, eighth by
impurity and tied for second by permutation. With 36 validation rows the standard
deviations are as large as several of the means, so read the top of this ranking
and ignore the bottom - and read neither one as a causal effect.

## Part 5: XGBoost

XGBoost builds trees in sequence, each one aimed at what the ensemble so far
still gets wrong. It follows the same fit/predict pattern.

```python
xgb_model = xgb.XGBClassifier(
    n_estimators=100, max_depth=3, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1,
)
xgb_model.fit(X_train, y_train)
xgb_acc = accuracy_score(y_valid, xgb_model.predict(X_valid))
print(f"XGBoost validation accuracy: {xgb_acc:.4f}")

importance_comparison = pd.merge(
    rf_importance[['feature', 'impurity_importance']],
    pd.DataFrame({'feature': feature_cols, 'xgboost': xgb_model.feature_importances_.round(3)}),
    on='feature',
)
print("\n=== Impurity importance, two libraries ===")
print(importance_comparison.head(6).to_string(index=False))

alt.Chart(
    importance_comparison.melt(id_vars='feature', var_name='model', value_name='importance')
).mark_bar().encode(
    x=alt.X('importance:Q', title='Feature importance'),
    y=alt.Y('feature:N', title='Feature', sort='-x'),
    color='model:N',
    column='model:N',
).properties(width=200, height=300)
```

XGBoost leans hardest on `color_intensity` (0.423), which the random forest ranks
sixth at 0.055. Each library computes importance its own way, so compare rankings,
not numbers.

## Part 6: Early Stopping

Early stopping watches validation performance after every boosting round and stops
when it stops improving, so you can set a generous round budget without paying for
the rounds that only overfit. Because validation now helps *fit* this model, its
validation score is no longer on equal footing with the others - that is a cost of
early stopping, not a free win.

```python
# In XGBoost 2.0+, early_stopping_rounds is passed to the constructor
xgb_early_stop = xgb.XGBClassifier(
    n_estimators=500,           # Set high; early stopping decides where to stop
    max_depth=3, learning_rate=0.1,
    early_stopping_rounds=10,
    random_state=42, n_jobs=-1,
)
xgb_early_stop.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
xgb_early_acc = accuracy_score(y_valid, xgb_early_stop.predict(X_valid))

print(f"Best iteration: {xgb_early_stop.best_iteration} (of a 500-round budget)")
print(f"Best validation loss: {xgb_early_stop.best_score:.4f}")
print(f"XGBoost, 100 fixed rounds - validation accuracy: {xgb_acc:.4f}")
print(f"XGBoost, early stopped    - validation accuracy: {xgb_early_acc:.4f}")
```

```text
Best iteration: 304 (of a 500-round budget)
Best validation loss: 0.0570
XGBoost, 100 fixed rounds - validation accuracy: 0.9722
XGBoost, early stopped    - validation accuracy: 1.0000
```

On a table this small the validation loss keeps creeping down for hundreds of
rounds, so early stopping saves part of the budget rather than most of it. The
early-stopped model is the only candidate to get all 36 validation rows right -
and those same 36 rows chose its round count. Part 10 comes back to that.

## Part 7: Build and Compile a Neural Network

A `Sequential` model is a straight stack of layers. Two hidden ReLU layers
followed by a single sigmoid unit is a standard starting point for a yes/no
target. Compiling attaches three choices: the **optimizer** that updates the
weights (Adam), the **loss** it minimises (`binary_crossentropy` for a yes/no
target), and the **metrics** reported each epoch.

```python
n_features = X_train.shape[1]  # 13 wine measurements

model = keras.Sequential([
    keras.layers.Input(shape=(n_features,)),
    keras.layers.Dense(64, activation='relu', name='hidden1'),
    keras.layers.Dense(32, activation='relu', name='hidden2'),
    keras.layers.Dense(1, activation='sigmoid', name='output'),
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
```

The summary reports 3,009 weights - for 106 training rows. That ratio is why the
loss curves in the next part are worth watching.

## Part 8: Train and Read the Curves

An **epoch** is one pass through the training rows; `batch_size=32` means the
weights are updated after every 32 rows. Passing `validation_data` scores the
validation rows after every epoch, which is what turns training into something you
can diagnose.

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

valid_loss, valid_accuracy = model.evaluate(X_valid_scaled, y_valid, verbose=0)
print(f"\nValidation loss: {valid_loss:.4f}")
print(f"Validation accuracy: {valid_accuracy:.4f}")
print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
print(confusion_matrix(y_valid, (model.predict(X_valid_scaled, verbose=0) > 0.5).astype(int)))
```

```text
 accuracy   loss  val_accuracy  val_loss  epoch
      1.0 0.0083        0.9722    0.0885     48
      1.0 0.0080        0.9722    0.0880     49
      1.0 0.0076        0.9722    0.0875     50

Validation loss: 0.0875
Validation accuracy: 0.9722

Confusion matrix [[TN, FP], [FN, TP]]:
[[24  0]
 [ 1 11]]
```

```python
loss_long = history_df.melt(
    id_vars='epoch', value_vars=['loss', 'val_loss'],
    var_name='curve', value_name='cross_entropy',
)

alt.Chart(loss_long).mark_line(point=True).encode(
    x=alt.X('epoch:Q', title='Epoch'),
    y=alt.Y('cross_entropy:Q', title='Binary cross-entropy loss'),
    color='curve:N',
).properties(width=400, height=250, title='Training and validation loss')
```

**What to look for:** training loss falling is the model learning. Validation loss
falling with it means the network is generalising; validation loss turning back up
while training loss keeps falling is overfitting, and it is the signal early
stopping watches for. Here training loss falls below
0.01 while validation loss settles roughly ten times higher - the gap you
expect when 3,009 weights meet 106 rows.

## Part 9: Two Variants

Depth, width, and regularization are settings to validate, not upgrades. Train one
variant with more capacity and one with **dropout** (randomly masking 30% of a
layer's outputs during training; all units are active at prediction time) plus an
L2 penalty on the weights. Same epochs, same batch size, so the comparison is
fair.

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

print(pd.DataFrame({
    'Network': ['64-32', '128-64-32-16', '64-32 + dropout/L2'],
    'Weights': [model.count_params(), model_deep.count_params(), model_reg.count_params()],
    'Validation loss': [valid_loss, deep_loss, reg_loss],
    'Validation accuracy': [valid_accuracy, deep_acc, reg_acc],
}).round(4).to_string(index=False))
```

```text
           Network  Weights  Validation loss  Validation accuracy
             64-32     3009           0.0875               0.9722
      128-64-32-16    12673           0.1174               0.9722
64-32 + dropout/L2     3009           0.3443               0.9722
```

All three get the same 35 of 36 validation rows right. Four times the weights
changes nothing measurable, and the regularized network is *worse* on validation
loss - the penalty and the dropped units cost it confidence it did not need to
give up. Plot the two loss curves to see where that happens.

```python
val_loss_curves = pd.DataFrame({
    'epoch': history_df['epoch'],
    'plain 64-32': history_df['val_loss'],
    'with dropout/L2': pd.DataFrame(history_reg.history)['val_loss'],
}).melt(id_vars='epoch', var_name='network', value_name='val_loss')

alt.Chart(val_loss_curves).mark_line(point=True).encode(
    x=alt.X('epoch:Q', title='Epoch'),
    y=alt.Y('val_loss:Q', title='Validation loss'),
    color='network:N',
).properties(width=400, height=250, title='Validation loss: plain vs regularized')
```

Regularization is a setting to validate, not an upgrade: on a table this small
the plain network was never overfitting enough for it to help.

## Part 10: Compare Every Candidate

One validation set, one metric, every model this notebook fitted - including the
early-stopped XGBoost, which is easy to leave out of a table precisely because it
is the outlier.

```python
comparison = pd.DataFrame({
    'Model': ['Majority-class baseline', 'Logistic regression', 'Random forest',
              'XGBoost (100 rounds)', 'XGBoost (early stopped)', 'Neural net 64-32',
              'Neural net 128-64-32-16', 'Neural net 64-32 + dropout/L2'],
    'Validation accuracy': [baseline_acc, lr_acc, rf_acc, xgb_acc, xgb_early_acc,
                            valid_accuracy, deep_acc, reg_acc],
})
comparison['Correct of 36'] = (comparison['Validation accuracy'] * len(y_valid)).round().astype(int)
print(comparison.to_string(index=False))

alt.Chart(comparison).mark_bar().encode(
    x=alt.X('Model:N', title='', sort='-y'),
    y=alt.Y('Validation accuracy:Q', scale=alt.Scale(domain=[0.6, 1.0])),
).properties(width=400, height=300)
```

```text
                        Model  Validation accuracy  Correct of 36
      Majority-class baseline             0.666667             24
          Logistic regression             0.972222             35
                Random forest             0.972222             35
         XGBoost (100 rounds)             0.972222             35
      XGBoost (early stopped)             1.000000             36
             Neural net 64-32             0.972222             35
      Neural net 128-64-32-16             0.972222             35
Neural net 64-32 + dropout/L2             0.972222             35
```

Every model clears the baseline by a wide margin, and six of the seven land on the
same score, 35 of 36. Identical scores do not mean identical predictions, so check
which row each one gets wrong. TensorFlow prints a `tf.function retracing` warning
when three networks predict one after another; it is noise, not a failure.

```python
valid_predictions = {
    'Logistic regression': lr.predict(X_valid_scaled),
    'Random forest': rf_model.predict(X_valid),
    'XGBoost (100 rounds)': xgb_model.predict(X_valid),
    'XGBoost (early stopped)': xgb_early_stop.predict(X_valid),
    'Neural net 64-32': (model.predict(X_valid_scaled, verbose=0) > 0.5).astype(int).flatten(),
    'Neural net 128-64-32-16': (model_deep.predict(X_valid_scaled, verbose=0) > 0.5).astype(int).flatten(),
    'Neural net 64-32 + dropout/L2': (model_reg.predict(X_valid_scaled, verbose=0) > 0.5).astype(int).flatten(),
}

row_numbers = np.arange(len(y_valid))
for name, pred in valid_predictions.items():
    missed = row_numbers[pred != y_valid]
    print(f"{name:30s} misses validation rows {missed.tolist()}")
```

```text
Logistic regression            misses validation rows [30]
Random forest                  misses validation rows [30]
XGBoost (100 rounds)           misses validation rows [18]
XGBoost (early stopped)        misses validation rows []
Neural net 64-32               misses validation rows [30]
Neural net 128-64-32-16        misses validation rows [30]
Neural net 64-32 + dropout/L2  misses validation rows [30]
```

Five of the seven miss the same row, number 30 - a cultivar-0 wine that logistic
regression, the forest, and all three networks all call negative. The 100-round
XGBoost gets row 30 right and misses row 18 instead, so it earns its 35/36 on a
different set of rows. The early-stopped XGBoost misses neither, which is worth
exactly one row - and those same 36 rows chose its round count, so its 36/36 is
the most optimistic number in the table, not the most trustworthy. Two models that
tie at 35/36 while disagreeing about which wine they get wrong are not the same
model; 36 rows simply cannot tell you which one to prefer.

Apply Part 2's rule. The best and the runner-up differ by one validation row, so
they are tied, and the tie goes to the simplest candidate: **logistic
regression**, thirteen coefficients and an intercept, fitted in milliseconds. That
is the honest reading of 36 validation rows - not that deep learning failed, but
that this dataset is far too small to tell these families apart. With thousands of
rows and messier features the ranking could look completely different, which is
why the rule is about *this* evidence.

## Part 11: Freeze, Then Test Once

Freezing means the features, preprocessing, and settings stop changing. Refit the
frozen configuration on the combined training and validation rows - more data, no
new choices - and score it on the test set exactly once.

```python
final_scaler = StandardScaler()
X_train_valid_scaled = final_scaler.fit_transform(X_train_valid)
X_test_final_scaled = final_scaler.transform(X_test)

final_model = LogisticRegression(max_iter=1000, random_state=42)
final_model.fit(X_train_valid_scaled, y_train_valid)

test_pred = final_model.predict(X_test_final_scaled)
print("=== Final test performance: frozen logistic regression ===")
print(f"Test accuracy: {accuracy_score(y_test, test_pred):.4f}")
print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
print(confusion_matrix(y_test, test_pred))
```

```text
=== Final test performance: frozen logistic regression ===
Test accuracy: 0.9722

Confusion matrix [[TN, FP], [FN, TP]]:
[[23  1]
 [ 0 12]]
```

One false positive, no missed cultivar-0 wines, and the number goes in the report
as it came out. Going back now to try the early-stopped XGBoost on these rows
would turn the test set into a second validation set.

## Key Takeaways

1. **Random Forest**: Handles non-linear relationships and interactions with no scaling
2. **Two importance views**: `feature_importances_` is the training fit; `permutation_importance` is held-out reliance, and they can disagree
3. **XGBoost**: Trees in sequence, each fixing what the ensemble still gets wrong
4. **Early stopping**: A generous round budget plus validation, at the cost of an optimistic validation score
5. **Data scaling**: Always scale features for neural networks (and for logistic regression)
6. **Loss curves**: Watch training and validation loss together, not accuracy alone
7. **Architecture and regularization**: Settings to validate, not upgrades
8. **A rule fixed in advance**: Decide how you will pick before you see the table
9. **Deep learning isn't always better**: For small tabular data, the simplest model is often indistinguishable from the most complex one

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
- Explore PyTorch for more flexibility
