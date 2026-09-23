# Demo 2: Honest Prediction with scikit-learn

## Learning Objectives
- Master the scikit-learn fit/predict pattern
- Split rows into training, validation, and test roles
- Beat a mean baseline before believing any model
- Keep preprocessing honest with `Pipeline` and `ColumnTransformer`
- Compare candidates with MAE, RMSE, and R² on validation rows
- Read what a fitted pipeline relies on with permutation importance
- Score a yes/no decision with accuracy, precision, and recall
- Freeze one candidate and evaluate it on the test set exactly once

## Setup

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import altair as alt

np.random.seed(42)
```

## Part 1: Load Real Dataset

Let's use the California Housing dataset - a real-world dataset from the 1990 US Census. This is the same dataset used in Demo 1, but now we'll apply machine learning techniques to it.

```python
# Load California Housing dataset from scikit-learn
from sklearn.datasets import fetch_california_housing

# Fetch the dataset
housing_data = fetch_california_housing(as_frame=True)
df = housing_data.frame

# Rename target for clarity
df = df.rename(columns={'MedHouseVal': 'house_value'})

# The dataset contains:
# - MedInc: median income in block group
# - HouseAge: median house age in block group
# - AveRooms: average number of rooms per household
# - AveBedrms: average number of bedrooms per household
# - Population: block group population
# - AveOccup: average number of household members
# - Latitude: block group latitude
# - Longitude: block group longitude
# - house_value: median house value (target, in hundreds of thousands of dollars)

print("Dataset shape:", df.shape)
print("\nFeature names:", housing_data.feature_names)
print("\nFirst few rows:")
print(df.head())
print("\nSummary statistics:")
print(df.describe())
```

Every number reported below is in the target's own units: **hundreds of thousands of dollars**. An MAE of 0.53 means the typical miss is about $53,000.

## Part 2: Train/Validation/Test Split

The golden rule: never evaluate on data the model has seen during training!

Before we can train any machine learning model, we need to split our data. The validation set selects models and tuning choices; the test set stays untouched until one final evaluation of the frozen choice. These census rows have no time order, so a seeded random split is fair here; Demo 1's clinic visits needed a chronological one.

```python
# Prepare features and target
feature_cols = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                'Population', 'AveOccup', 'Latitude', 'Longitude']
X = df[feature_cols]
y = df['house_value']

# Reserve 20% as an untouched final test set, then split the remainder for validation.
X_train_valid, X_test, y_train_valid, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_valid, y_train_valid, test_size=0.25, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_valid.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"\nTraining target statistics:")
print(y_train.describe())
print(f"\nValidation target statistics:")
print(y_valid.describe())
```

Expect 12,384 / 4,128 / 4,128 rows.

**Why split the data?**
- **Training set**: Used to teach the model patterns in the data
- **Validation set**: Used to compare candidate models and tune training choices
- **Test set**: Held untouched until the final, one-time evaluation
- **60/20/20 split**: A simple teaching split; proportions depend on dataset size
- **random_state=42**: Ensures reproducible splits (same random seed = same split)

## Part 3: Leakage-Safe Mixed-Type Preprocessing

The housing table is numeric, but many real tables mix numeric and categorical
predictors. This compact pattern fits every preprocessing step on training rows
and reuses it unchanged on validation or test rows. `ColumnTransformer` routes
columns to numeric and categorical branches; the categorical branch imputes
missing values and ignores categories it did not see during fitting.

```python
# A small standalone example of the pattern used before model fitting.
mixed_train = pd.DataFrame({
    'income': [3.2, 5.1, np.nan, 2.8],
    'rooms': [4.0, 6.5, 5.2, 3.8],
    'region': ['north', 'south', 'north', 'south'],
})
mixed_target = pd.Series([1.2, 2.4, 1.8, 1.0], name='target')
mixed_valid = pd.DataFrame({
    'income': [4.4, 3.0],
    'rooms': [5.9, np.nan],
    'region': ['central', 'north'],  # 'central' is unseen during fitting
})

numeric_branch = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scale', StandardScaler()),
])
categorical_branch = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('one_hot', OneHotEncoder(handle_unknown='ignore')),
])
mixed_preprocess = ColumnTransformer([
    ('numeric', numeric_branch, ['income', 'rooms']),
    ('categorical', categorical_branch, ['region']),
])
mixed_model = Pipeline([
    ('preprocess', mixed_preprocess),
    ('regressor', Ridge(alpha=1.0)),
])
mixed_model.fit(mixed_train, mixed_target)
mixed_predictions = mixed_model.predict(mixed_valid)
print('Predictions for mixed-type validation rows:', mixed_predictions)
```

`SimpleImputer` is one documented option when missing predictors need a value;
it is not automatically required. The important boundary is that the imputer,
encoder, and scaler learn from training data only. In cross-validation, put the
whole pipeline inside the candidate being evaluated so each fold has the same
protection.

## Part 4: A Baseline to Beat

Before celebrating any model, ask what a guess would score. `DummyRegressor(strategy='mean')` predicts the training mean for every row, and it defines the bar: a model that cannot beat it has learned nothing useful.

Every candidate below is scored the same way, so the comparison is fair.

```python
results = []

def score_candidate(name, fitted_model):
    """Score a fitted model on the validation rows and record the result."""
    predictions = fitted_model.predict(X_valid)
    row = {
        'model': name,
        'valid_MAE': mean_absolute_error(y_valid, predictions),
        'valid_RMSE': np.sqrt(mean_squared_error(y_valid, predictions)),
        'valid_R2': r2_score(y_valid, predictions),
    }
    results.append(row)
    print(f"{name}: MAE={row['valid_MAE']:.3f}  RMSE={row['valid_RMSE']:.3f}  R²={row['valid_R2']:.3f}")

baseline = DummyRegressor(strategy='mean')
baseline.fit(X_train, y_train)

print(f"Training mean: {y_train.mean():.3f}")
print(f"First three baseline predictions: {baseline.predict(X_valid)[:3].round(3)}")
score_candidate('mean baseline', baseline)
```

```text
Training mean: 2.068
First three baseline predictions: [2.068 2.068 2.068]
mean baseline: MAE=0.925  RMSE=1.172  R²=-0.000
```

The baseline's R² is a hair below 0 because it predicts the _training_ mean (2.068), which misses the validation rows' own mean (2.084). Anything with MAE above 0.925 is worse than guessing.

## Part 5: Linear Regression with scikit-learn

scikit-learn's API is consistent across all models: create, fit, predict.

The scikit-learn workflow is beautifully simple: create the model, fit it to training data, then make predictions. Wrapping the model in a `Pipeline` with `StandardScaler` keeps that simplicity and guarantees the scaler learns its means and standard deviations from training rows only.

```python
# Create and fit a scaled linear regression pipeline
linear = Pipeline([('scale', StandardScaler()), ('model', LinearRegression())])
linear.fit(X_train, y_train)

score_candidate('linear regression', linear)

# Training performance, for comparison with validation performance
train_r2 = r2_score(y_train, linear.predict(X_train))
print(f"Training R²: {train_r2:.3f}")

print("\nCoefficients (on scaled features, so they are comparable):")
coef_df = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': linear.named_steps['model'].coef_.round(3)
})
print(coef_df)
print(f"\nIntercept: {linear.named_steps['model'].intercept_:.3f}")
```

**Understanding the metrics:**
- **MAE (mean absolute error)**: The average size of a miss, in hundreds of thousands of dollars. Lower is better.
- **RMSE (root mean squared error)**: Like MAE, but large misses count extra, so it is never smaller than MAE.
- **R²**: How much better than predicting the mean of these rows. 1 is perfect, 0 ties the mean, and **negative is worse than the mean**.
- **Training vs validation**: If training performance is much better than validation, the model may be overfitting. Here they are close, so this model is not memorizing.
- **Coefficients**: Because the pipeline scaled the features first, these are comparable across columns.

## Part 6: Regularized Linear Models

Regularization adds a penalty on the coefficients that can reduce overfitting. Think of it as a "simplicity penalty" - the model is rewarded for using smaller coefficients.

**Ridge (L2) regularization** shrinks all coefficients toward zero but doesn't eliminate them. **Lasso (L1) regularization** can set coefficients to exactly zero, which selects features. Both penalties treat every coefficient alike, so the features must be scaled first - which is exactly what the pipeline does.

```python
# Ridge Regression (L2 regularization)
ridge = Pipeline([('scale', StandardScaler()), ('model', Ridge(alpha=10.0))])
ridge.fit(X_train, y_train)
score_candidate('ridge (alpha=10)', ridge)

# Lasso Regression (L1 regularization - can zero out coefficients)
lasso = Pipeline([('scale', StandardScaler()), ('model', Lasso(alpha=0.1))])
lasso.fit(X_train, y_train)
score_candidate('lasso (alpha=0.1)', lasso)

# Compare coefficients
coef_comparison = pd.DataFrame({
    'feature': feature_cols,
    'linear': linear.named_steps['model'].coef_.round(3),
    'ridge': ridge.named_steps['model'].coef_.round(3),
    'lasso': lasso.named_steps['model'].coef_.round(3),
})
print("\n=== Coefficient Comparison (scaled features) ===")
print(coef_comparison)

# Lasso can zero out features (feature selection)
nonzero = int((lasso.named_steps['model'].coef_ != 0).sum())
print(f"\nFeatures kept by Lasso (non-zero coefficients): {nonzero} of {len(feature_cols)}")
```

```text
ridge (alpha=10): MAE=0.533  RMSE=0.728  R²=0.614
lasso (alpha=0.1): MAE=0.626  RMSE=0.826  R²=0.503
```

**What the numbers say here:** with 12,384 training rows and only 8 features, there is little overfitting for a penalty to fix, so Ridge lands on top of plain linear regression. Lasso at `alpha=0.1` keeps 3 of 8 features - it zeroes `Longitude` outright and shrinks `Latitude` to -0.012 - and pays for dropping location with a clearly worse validation score. Regularization is a setting to validate, not a free improvement.

**When to reach for it:**
- **Many features**: helpful when features outnumber observations
- **Multicollinearity**: correlated features make unpenalized coefficients unstable
- **Feature selection**: Lasso reports a shorter feature list
- **Overfitting**: a large train-validation gap is the symptom to look for

## Part 7: Model Comparison

Let's put every candidate side by side. One table, one metric set, one validation set.

```python
comparison = pd.DataFrame(results).round(4)
print("=== Validation Comparison ===")
print(comparison.to_string(index=False))

# Visualize comparison
comparison_long = comparison.melt(
    id_vars='model',
    value_vars=['valid_MAE', 'valid_RMSE'],
    var_name='metric',
    value_name='error'
)

alt.Chart(comparison_long).mark_bar().encode(
    x=alt.X('model:N', title='Model', sort='-y'),
    y=alt.Y('error:Q', title='Validation error (hundreds of thousands of dollars)'),
    color='metric:N',
    column='metric:N'
).properties(
    width=150,
    height=300
)
```

```text
            model  valid_MAE  valid_RMSE  valid_R2
    mean baseline     0.9248      1.1719   -0.0002
linear regression     0.5333      0.7278    0.6142
 ridge (alpha=10)     0.5333      0.7278    0.6143
lasso (alpha=0.1)     0.6257      0.8262    0.5029
```

Both linear models cut the baseline's typical miss almost in half. Ridge and plain linear regression then tie on MAE and separate by 0.0001 of R², a gap far smaller than the difference another random split would produce, so treat them as tied. A tie-break rule fixed _before_ looking - prefer the penalized model, and record its `alpha` - selects the Ridge pipeline as the candidate to freeze.

## Part 8: Validation Prediction Visualization

Visualize how well the selected model predicts house values.

```python
# Use validation predictions while the test set remains sealed.
ridge_valid_pred = ridge.predict(X_valid)
pred_df = pd.DataFrame({
    'actual': y_valid.values,
    'predicted': ridge_valid_pred,
    'error': y_valid.values - ridge_valid_pred
})

# Scatter plot: actual vs predicted
scatter = alt.Chart(pred_df).mark_circle(opacity=0.5).encode(
    x=alt.X('actual:Q', title='Actual house value (hundreds of thousands)'),
    y=alt.Y('predicted:Q', title='Predicted house value (hundreds of thousands)'),
    color=alt.Color('error:Q', scale=alt.Scale(scheme='redblue', domainMid=0), 
                    title='Error')
).properties(
    width=400,
    height=400
)

# Add perfect prediction line (y=x)
perfect_line = alt.Chart(pd.DataFrame({'x': [pred_df['actual'].min(), pred_df['actual'].max()]})).mark_line(
    color='red', strokeDash=[5, 5]
).encode(
    x='x:Q',
    y='x:Q'
)

(scatter + perfect_line).resolve_scale(color='independent')
```

```python
# Residual plot (errors vs predicted)
residual_chart = alt.Chart(pred_df).mark_circle(opacity=0.5).encode(
    x=alt.X('predicted:Q', title='Predicted house value (hundreds of thousands)'),
    y=alt.Y('error:Q', title='Residual (actual - predicted)'),
    color=alt.Color('error:Q', scale=alt.Scale(scheme='redblue', domainMid=0))
).properties(
    width=400,
    height=300
)

# Add zero line
zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='red', strokeDash=[5, 5]).encode(y='y:Q')

residual_chart + zero_line
```

**What to look for:**
- **Scatter plot**: Points should cluster around the red diagonal line (perfect predictions)
- **Residual plot**: Errors should be randomly distributed around zero (no patterns)
- **The flat band at 5.0**: the census capped house values there, so the model cannot follow the most expensive blocks

## Part 9: What Does the Model Rely On?

A metric says how well the pipeline predicts; it does not say which columns carry the prediction. Permutation importance shuffles one column of the validation set at a time and measures how much the score gets worse. It works on any fitted estimator, including a pipeline, because it only needs `predict`.

We use the validation set, not the final test set, and keep MAE as the evaluation measure. scikit-learn orients scorers so higher is better, so MAE uses the negative-MAE scorer; a positive result below is the increase in validation MAE after shuffling.

```python
# Measure predictive reliance on validation data with reproducible shuffles.
permutation_result = permutation_importance(
    ridge,
    X_valid,
    y_valid,
    scoring='neg_mean_absolute_error',
    n_repeats=10,
    random_state=42,
    n_jobs=-1,
)

permutation_df = pd.DataFrame({
    'feature': feature_cols,
    'validation_mae_increase': permutation_result.importances_mean.round(3),
    'repeat_std': permutation_result.importances_std.round(3),
}).sort_values('validation_mae_increase', ascending=False)

print("=== Ridge Permutation Importance (Validation MAE Increase) ===")
print(permutation_df.to_string(index=False))
```

```text
   feature  validation_mae_increase  repeat_std
  Latitude                    0.629       0.011
 Longitude                    0.605       0.007
    MedInc                    0.566       0.009
 AveBedrms                    0.068       0.004
  AveRooms                    0.061       0.004
  HouseAge                    0.011       0.002
  AveOccup                    0.003       0.000
Population                   -0.000       0.000
```

Location and income carry the model; shuffling `Population` costs nothing at all. Two cautions: correlated features can substitute for one another, so shuffling either one may show little damage or divide importance between them, and these values describe this fitted model's reliance under shuffling. They are not causal effects, and they do not establish that changing a feature would change house values.

## Part 10: When the Prediction Becomes a Yes/No Decision

Predictions usually end in a decision. Suppose a housing programme wants to flag
block groups whose median value is above 3.5 (that is, $350,000) so an analyst can
review them. Turning the frozen model's numeric prediction into a flag turns a
regression problem into a yes/no one, and yes/no results need different metrics.

The comparison here is the flag against the laziest possible policy: never flag
anything. Watch what that does to accuracy.

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

threshold = 3.5
actual_expensive = (y_valid.values > threshold).astype(int)
ridge_flag = (ridge_valid_pred > threshold).astype(int)
never_flag = np.zeros_like(actual_expensive)

print(f"Expensive block groups in validation: {actual_expensive.sum()} of {len(actual_expensive)}")

binary_rows = []
for name, predicted in [('ridge_flag', ridge_flag), ('never_flag', never_flag)]:
    binary_rows.append({
        'approach': name,
        'accuracy': accuracy_score(actual_expensive, predicted),
        'precision': precision_score(actual_expensive, predicted, zero_division=0),
        'recall': recall_score(actual_expensive, predicted),
    })
print(pd.DataFrame(binary_rows).round(3).to_string(index=False))

print("\nRidge flag, confusion matrix [[TN, FP], [FN, TP]]:")
print(confusion_matrix(actual_expensive, ridge_flag))
```

```text
Expensive block groups in validation: 520 of 4128
  approach  accuracy  precision  recall
ridge_flag     0.917      0.876   0.394
never_flag     0.874      0.000   0.000

Ridge flag, confusion matrix [[TN, FP], [FN, TP]]:
[[3579   29]
 [ 315  205]]
```

Never flagging anything is 87.4% accurate and finds nothing: when positives are
rare, accuracy mostly measures how rare they are. `zero_division=0` is what keeps
`precision_score` from warning about the 0/0 that policy produces. The model's
flags are trustworthy when it raises them (precision 0.876) but it stays silent on
315 of the 520 expensive block groups (recall 0.394). Which of those two failures
costs more is a question about the programme, not about the model - and the answer
decides whether 3.5 is the right threshold.

## Part 11: One Final Test Evaluation

Validation chose the Ridge pipeline, so **freeze** it: same steps, same `alpha`, no more changes. Refitting the frozen configuration on the combined training and validation rows gives it more data without changing any setting. This is the first and only point at which the test set is used.

```python
# Record exactly what is being frozen
print("Frozen steps:", [name for name, _ in ridge.steps])
print("Frozen settings:", ridge.named_steps['model'].get_params())

final_model = Pipeline([('scale', StandardScaler()), ('model', Ridge(alpha=10.0))])
final_model.fit(X_train_valid, y_train_valid)

final_test_pred = final_model.predict(X_test)
print("\n=== Final Test Performance: frozen Ridge pipeline ===")
print(f"Test MAE: {mean_absolute_error(y_test, final_test_pred):.3f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, final_test_pred)):.3f}")
print(f"Test R²: {r2_score(y_test, final_test_pred):.3f}")
```

```text
=== Final Test Performance: frozen Ridge pipeline ===
Test MAE: 0.533
Test RMSE: 0.745
Test R²: 0.576
```

The test R² (0.576) is a little below the validation R² (0.614). That is the normal cost of having _chosen_ on validation rows, and it is the number that gets reported - no going back to try another `alpha`.

## Key Takeaways

1. **scikit-learn API**: Consistent fit/predict pattern across all models
2. **Train/validation/test split**: Select with validation; report the test result once
3. **Baseline first**: `DummyRegressor` sets the bar every model has to clear
4. **Pipelines**: Scaling and imputation belong inside the model so they learn from training rows only
5. **Metrics**: MAE, RMSE, and R² on validation rows, computed the same way for every candidate
6. **Regularization**: Ridge and Lasso are settings to validate, not free improvements
7. **Permutation importance**: Held-out reliance for any fitted estimator, not a causal claim
8. **Yes/no decisions**: When positives are rare, accuracy flatters a policy that finds nothing; read precision and recall
9. **Freeze, then test once**: Record the settings, refit, and report the test number as it comes

## Next Steps

- Experiment with hyperparameter tuning (GridSearchCV)
- Try other scikit-learn models (SVM, KNN)
- Learn about cross-validation for better model evaluation
- Demo 3 takes the same workflow to trees, boosting, and neural networks
