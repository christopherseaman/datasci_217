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

# Demo 2: Honest Prediction with scikit-learn

Demo 1 asked how disease progression _relates_ to BMI in 442 diabetes patients. This demo asks a prediction question about the same records: from a new patient's baseline measurements, how close can we get to their progression score one year later? You split the patients into training, validation, and test rows, set a baseline to beat, write down a selection rule, compare linear pipelines, read what the chosen one relies on, turn its predictions into a yes/no flag, and evaluate it on the test rows exactly once. Everything here comes from Lecture 10 up to the second demo break, plus Lectures 01 to 09. The records are real and de-identified.

**How to run:** open this notebook in Colab from the lecture page's Colab link, or locally in VS Code with the kernel set to a `.venv` made by `uv venv --seed` and `uv pip install -r requirements.txt` in this folder (Lecture 03). Run the cells from top to bottom; after each step, an **Expect** line says what you should see. Colab does not save your changes back to GitHub; use **File → Save a copy in Drive** to keep them. Tested 2026-09-25 with Python 3.13, pandas 3.0.5, NumPy 2.3.3, scikit-learn 1.9.0, and matplotlib 3.11.1; the whole notebook runs in a few seconds.

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
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

print('pandas', pd.__version__)
print('scikit-learn', sklearn.__version__)
```

**Expect:** `pandas 3.0.5` and `scikit-learn 1.9.0` (Colab may show an older scikit-learn; that is fine).

## 1. Load the diabetes records

The same table as Demo 1: ten baseline measurements per patient (age, sex, BMI, average blood pressure, and six blood tests) and the progression score one year later, which runs from 25 to 346. Every error below is in those score points.

```python
diabetes = load_diabetes(scaled=False, as_frame=True).frame
diabetes = diabetes.rename(columns={'s1': 'tc', 's2': 'ldl', 's3': 'hdl', 's4': 'tch',
                                    's5': 'ltg', 's6': 'glu', 'target': 'progression'})
feature_cols = ['age', 'sex', 'bmi', 'bp', 'tc', 'ldl', 'hdl', 'tch', 'ltg', 'glu']
X = diabetes[feature_cols]
y = diabetes['progression']
print(X.shape, y.shape)
```

**Expect:** `(442, 10) (442,)`: ten feature columns in `X` and one target column in `y`.

## 2. Train, validation, and test rows

These patients were all measured at one baseline and have no time order, so a seeded random split is fair here; Demo 1's weekly readings needed a chronological one. Set the test rows aside first, then split the rest into training and validation rows.

```python
# Reserve 20% as the untouched test set, then take 25% of the rest for validation.
X_train_valid, X_test, y_train_valid, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_valid, y_train_valid, test_size=0.25, random_state=42)

print(f"Training rows:   {len(X_train)}")
print(f"Validation rows: {len(X_valid)}")
print(f"Test rows:       {len(X_test)}")
print(f"Mean progression, training: {y_train.mean():.1f}, validation: {y_valid.mean():.1f}")
```

**Expect:**

```text
Training rows:   264
Validation rows: 89
Test rows:       89
Mean progression, training: 149.8, validation: 165.4
```

A 60/20/20 split. By chance, the validation patients progressed more on average than the training patients; Part 4 shows what that does to a baseline.

## 3. Preprocessing inside the pipeline

The diabetes table is all numbers with no gaps, but many clinic tables mix numbers with categories and have missing values. This small standalone table shows the pattern: `ColumnTransformer` sends the numeric columns to an imputer and a scaler, and the site column to an imputer and a one-hot encoder. Every step learns from the training rows only, and a site it never saw becomes all zeros instead of an error.

```python
clinic_train = pd.DataFrame({
    'age': [54, 61, 47, 70],
    'ldl': [130.0, np.nan, 110.0, 145.0],   # mg/dL; one missing
    'site': ['north', 'south', 'north', 'south'],
})
clinic_sbp = pd.Series([138, 146, 129, 152])  # mmHg
clinic_valid = pd.DataFrame({
    'age': [58, 66],
    'ldl': [np.nan, 150.0],
    'site': ['west', 'north'],   # 'west' never appears in training
})

numeric = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
categorical = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                        ('one_hot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
preprocess = ColumnTransformer([('numeric', numeric, ['age', 'ldl']),
                                ('categorical', categorical, ['site'])])
clinic_model = Pipeline([('preprocess', preprocess), ('model', Ridge(alpha=1.0))])
clinic_model.fit(clinic_train, clinic_sbp)

print('What the model sees for the two validation rows:')
print(clinic_model.named_steps['preprocess'].transform(clinic_valid).round(2))
print('Predicted SBP:', clinic_model.predict(clinic_valid).round(1))
```

**Expect:**

```text
What the model sees for the two validation rows:
[[0.   0.1  0.   0.  ]
 [0.94 1.71 1.   0.  ]]
Predicted SBP: [141.6 147.7]
```

Read the columns as scaled age, scaled LDL, `site_north`, `site_south`. The first row's age, 58, equals the training mean, so it scales to 0.0; its missing LDL was filled with the training median, 130 mg/dL, which scales to 0.1; and its unseen site, `west`, became all zeros. The second row is a `north` patient, `[1, 0]`. `named_steps['preprocess']` reached inside the fitted pipeline to show this.

## 4. A baseline, and the rule for choosing

Before scoring any real model, ask what a guess would score. `DummyRegressor(strategy='mean')` predicts the training mean for every patient; a model that cannot beat it has learned nothing useful.

Write the selection rule down now, before any candidate is scored, so the comparison table cannot be read backwards to favor one model:

- The lowest validation MAE wins.
- With only 89 validation patients, candidates within 1 point of the best MAE count as tied.
- A tie goes to the candidate that uses the fewest features, because every feature is a measurement someone must collect for each new patient.

Every candidate is scored by the same function, so the comparison is fair.

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
    print(f"{name}: MAE={row['valid_MAE']:.2f}  RMSE={row['valid_RMSE']:.2f}  R²={row['valid_R2']:.3f}")

baseline = DummyRegressor(strategy='mean')
baseline.fit(X_train, y_train)
print(f"First three baseline predictions: {baseline.predict(X_valid)[:3].round(1)}")
score_candidate('mean baseline', baseline)
```

**Expect:**

```text
First three baseline predictions: [149.8 149.8 149.8]
mean baseline: MAE=64.13  RMSE=75.64  R²=-0.044
```

The baseline's R² is below 0 because it predicts the _training_ mean, 149.8, which misses these validation patients' own mean, 165.4. Any candidate with an MAE above 64.13 is worse than guessing.

## 5. A linear regression pipeline

Wrap the model in a `Pipeline` with `StandardScaler`, so the scaler learns each feature's mean and spread from the training rows only. Compare training and validation R² to check for overfitting.

```python
linear = Pipeline([('scale', StandardScaler()), ('model', LinearRegression())])
linear.fit(X_train, y_train)
score_candidate('linear regression', linear)

print(f"Training R²: {r2_score(y_train, linear.predict(X_train)):.3f}")
print("\nCoefficients on scaled features (score points per standard deviation):")
print(pd.Series(linear.named_steps['model'].coef_, index=feature_cols).round(1))
```

**Expect:**

```text
linear regression: MAE=42.94  RMSE=51.19  R²=0.522
Training R²: 0.520

Coefficients on scaled features (score points per standard deviation):
age     1.1
sex   -11.4
bmi    25.1
bp     15.2
tc    -51.1
ldl    27.5
hdl    10.8
tch    15.0
ltg    37.3
glu     3.8
dtype: float64
```

- **Better than guessing:** the typical miss drops from 64 points to 43.
- **No overfitting:** training R² (0.520) and validation R² (0.522) are nearly equal, so the model is not memorizing its 264 training rows.
- **Coefficients that cancel:** `tc` (-51.1) and `ldl` (+27.5) are large and opposite. Total cholesterol and LDL move together (their correlation is 0.90), so the unpenalized fit can trade a big negative on one against a big positive on the other. Neither number means much on its own.

## 6. Ridge and Lasso

Regularization adds a penalty that shrinks coefficients. Ridge (L2) shrinks them all toward zero; Lasso (L1) can set some to exactly zero, which drops those features. Both treat every coefficient alike, so the scaler stays in front of them in the pipeline. The `alpha` values here are fixed in advance, not tuned.

```python
ridge = Pipeline([('scale', StandardScaler()), ('model', Ridge(alpha=10.0))])
ridge.fit(X_train, y_train)
score_candidate('ridge (alpha=10)', ridge)

lasso = Pipeline([('scale', StandardScaler()), ('model', Lasso(alpha=2.0))])
lasso.fit(X_train, y_train)
score_candidate('lasso (alpha=2)', lasso)

coefficients = pd.DataFrame({
    'linear': linear.named_steps['model'].coef_,
    'ridge': ridge.named_steps['model'].coef_,
    'lasso': lasso.named_steps['model'].coef_,
}, index=feature_cols)
print(coefficients.round(1))
print(f"\nFeatures Lasso keeps: {(coefficients['lasso'] != 0).sum()} of {len(feature_cols)}")
```

**Expect:**

```text
ridge (alpha=10): MAE=42.62  RMSE=51.16  R²=0.522
lasso (alpha=2): MAE=42.45  RMSE=51.03  R²=0.525
     linear  ridge  lasso
age     1.1    1.6    0.0
sex   -11.4  -11.0   -7.9
bmi    25.1   25.1   25.5
bp     15.2   14.6   13.2
tc    -51.1  -11.7   -6.7
ldl    27.5   -2.9   -0.0
hdl    10.8   -5.8   -9.7
tch    15.0    9.3    0.0
ltg    37.3   22.5   23.7
glu     3.8    4.4    2.6

Features Lasso keeps: 7 of 10
```

Ridge pulls `tc` from -51.1 to -11.7 and `ldl` from 27.5 to -2.9: the penalty stops the two correlated columns from canceling each other with large opposite coefficients. Lasso goes further and sets `age`, `ldl`, and `tch` to exactly zero (`-0.0` is still zero). The strong predictors, `bmi`, `bp`, and `ltg`, barely move.

## 7. Compare the candidates and apply the rule

One table, one metric set, one validation set.

```python
comparison = pd.DataFrame(results)
comparison['features_used'] = [0, 10, 10, (coefficients['lasso'] != 0).sum()]
print(comparison.round(3).to_string(index=False))
```

**Expect:**

```text
            model  valid_MAE  valid_RMSE  valid_R2  features_used
    mean baseline     64.129      75.639    -0.044              0
linear regression     42.939      51.194     0.522             10
 ridge (alpha=10)     42.620      51.160     0.522             10
  lasso (alpha=2)     42.451      51.034     0.525              7
```

Apply Part 4's rule. The Lasso has the lowest MAE, 42.451. Linear regression (42.939) and Ridge (42.620) are within 1 point of it, so all three count as tied: on 89 patients a half-point gap is noise. The tie goes to the fewest features, which is the Lasso with 7 of 10. **Selected: the Lasso pipeline with `alpha=2`.** All three cut the baseline's typical miss by about a third.

## 8. Look at the validation predictions

Two plots of the selected pipeline's validation predictions, with Lecture 07's Axes methods: predicted against actual, where perfect predictions would sit on the dashed diagonal, and residuals against predictions, where we hope for a shapeless cloud around zero.

```python
lasso_valid_pred = lasso.predict(X_valid)
residuals = y_valid - lasso_valid_pred
print(f"Mean residual: {residuals.mean():.1f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(y_valid, lasso_valid_pred, alpha=0.6)
axes[0].plot([25, 346], [25, 346], color='gray', linestyle='--')
axes[0].set_xlabel('Actual progression')
axes[0].set_ylabel('Predicted progression')
axes[1].scatter(lasso_valid_pred, residuals, alpha=0.6)
axes[1].axhline(0, color='gray', linestyle='--')
axes[1].set_xlabel('Predicted progression')
axes[1].set_ylabel('Residual (actual - predicted)')
plt.show()
plt.close(fig)
```

**Expect:** `Mean residual: 14.2`, and two panels:

- **Left:** the points follow the diagonal but more flatly. The highest actual scores are predicted too low and the lowest too high, as expected from a model that explains about half the variation.
- **Right:** no curve and no funnel, but more points above the zero line than below. That is the 14.2-point average shift: these validation patients progressed more than the training mean suggested.

## 9. What does the pipeline rely on?

A metric says how well the pipeline predicts, not which columns carry the prediction. Permutation importance shuffles one validation column at a time and measures how much worse the MAE gets. scikit-learn scorers count higher as better, so MAE is `'neg_mean_absolute_error'`, and each result below is the increase in validation MAE after shuffling.

```python
permutation_result = permutation_importance(
    lasso, X_valid, y_valid,
    scoring='neg_mean_absolute_error', n_repeats=10, random_state=42,
)
importance = pd.DataFrame({
    'feature': feature_cols,
    'mae_increase': permutation_result.importances_mean,
    'std': permutation_result.importances_std,
}).sort_values('mae_increase', ascending=False)
print(importance.round(2).to_string(index=False))
```

**Expect:**

```text
feature  mae_increase  std
    bmi         10.50 1.98
    ltg         10.43 2.06
     bp          3.79 1.07
    hdl          2.90 1.01
     tc          0.78 0.59
    glu          0.42 0.27
    sex          0.23 0.49
    age          0.00 0.00
    ldl          0.00 0.00
    tch          0.00 0.00
```

BMI and triglycerides carry the pipeline: shuffling either one adds about 10.5 points to the validation MAE. `age`, `ldl`, and `tch` show exactly 0.00 because the Lasso set their coefficients to zero, so shuffling them cannot change a single prediction. Two cautions: correlated features such as `tc` and `ldl` can share or hide importance, and these numbers describe what this fitted pipeline relies on. They do not say that lowering a patient's triglycerides would slow their disease.

## 10. When the prediction becomes a yes/no decision

Predictions usually end in a decision. Suppose the clinic wants to schedule an early specialist review for patients whose progression score will exceed 200. Turning the selected pipeline's numeric prediction into a flag turns regression into classification, and yes/no results need different metrics. Compare the flag with the laziest possible policy: never flag anyone.

```python
threshold = 200
actual_high = (y_valid > threshold).astype(int)
lasso_flag = (lasso_valid_pred > threshold).astype(int)
never_flag = np.full(len(actual_high), 0)

print(f"Validation patients above {threshold}: {actual_high.sum()} of {len(actual_high)}")

binary_rows = []
for name, predicted in [('lasso_flag', lasso_flag), ('never_flag', never_flag)]:
    binary_rows.append({
        'policy': name,
        'accuracy': accuracy_score(actual_high, predicted),
        'precision': precision_score(actual_high, predicted, zero_division=0),
        'recall': recall_score(actual_high, predicted),
    })
print(pd.DataFrame(binary_rows).round(3).to_string(index=False))

print("\nLasso flag, confusion matrix [[TN, FP], [FN, TP]]:")
print(confusion_matrix(actual_high, lasso_flag))
```

**Expect:**

```text
Validation patients above 200: 29 of 89
    policy  accuracy  precision  recall
lasso_flag     0.753      0.706   0.414
never_flag     0.674      0.000   0.000

Lasso flag, confusion matrix [[TN, FP], [FN, TP]]:
[[55  5]
 [17 12]]
```

Never flagging anyone is right 67.4% of the time and finds nobody: accuracy mostly measures how common the negatives are. `zero_division=0` keeps `precision_score` from warning about the 0/0 that policy produces. The Lasso flag is right when it fires 12 times out of 17 (precision 0.706), but it misses 17 of the 29 patients who went above 200 (recall 0.414). Whether a missed patient costs more than an unneeded appointment is the clinic's call, and that answer decides whether 200 is the right point to flag at.

## 11. Freeze, then test once

Validation chose the Lasso pipeline, so **freeze** it: same steps, same `alpha`, no more changes. Record what is frozen, refit that configuration on the training and validation rows together (more data, no new choices), and score it on the test rows. This is the first and only time the test rows are used.

```python
print("Frozen steps:", [name for name, _ in lasso.steps])
print("Frozen alpha:", lasso.named_steps['model'].get_params()['alpha'])

final_model = Pipeline([('scale', StandardScaler()), ('model', Lasso(alpha=2.0))])
final_model.fit(X_train_valid, y_train_valid)

final_test_pred = final_model.predict(X_test)
print("\nFinal test performance, frozen Lasso pipeline:")
print(f"Test MAE:  {mean_absolute_error(y_test, final_test_pred):.2f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, final_test_pred)):.2f}")
print(f"Test R²:   {r2_score(y_test, final_test_pred):.3f}")
```

**Expect:**

```text
Frozen steps: ['scale', 'model']
Frozen alpha: 2.0

Final test performance, frozen Lasso pipeline:
Test MAE:  42.84
Test RMSE: 52.90
Test R²:   0.472
```

The test MAE (42.84) is close to the validation MAE (42.45); the test R² (0.472) is a little below the validation R² (0.525). A small drop is the normal cost of having _chosen_ on the validation rows, and these are also 89 different patients. This is the number that goes in the report, with no going back to try another `alpha`.
