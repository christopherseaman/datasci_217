# Q7: Modeling

**Phase 8:** Modeling  
**Points: 9 points**

**Focus:** Train multiple models, evaluate performance, compare models, extract feature importance.

**Lecture Reference:** See **Lecture 11, Notebook 4** (`11/demo/04_modeling_results.ipynb`), Phase 8 for examples of model training, evaluation, and comparison.

---

## Setup

```python
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import os

# Load prepared data from Q6
X_train = pd.read_csv('output/q6_X_train.csv')
X_test = pd.read_csv('output/q6_X_test.csv')
y_train = pd.read_csv('output/q6_y_train.csv').squeeze()  # Convert to Series
y_test = pd.read_csv('output/q6_y_test.csv').squeeze()

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
```

---

## Objective

Train multiple models, evaluate performance, compare models, and extract feature importance.

---

## ⚠️ Data Leakage Warning

If you see suspiciously perfect model performance, this likely indicates data leakage. Common warning signs:

**Warning Metrics:**
- **Perfect R² = 1.0000** (or very close, like 0.9999+)
- **Zero or near-zero RMSE/MAE** (e.g., RMSE < 0.01°C for temperature prediction)
- **Train and test performance nearly identical** (difference < 0.01)
- **Unrealistic precision**: Errors smaller than measurement precision (e.g., < 0.1°C for temperature sensors)
- **Feature correlation > 0.99** with target (check correlations between features and target)

**Common Causes:**
- **Circular prediction logic**: Using rolling windows of the target variable to predict itself
  - Example: Using `air_temp_rolling_7h` to predict `Air Temperature`
  - This is like predicting temperature from smoothed temperature - circular reasoning!
- **Features nearly identical to target**: Any feature with correlation > 0.99 with the target
- **Including target variable directly**: Accidentally including the target in features

**How to Check:**
- Calculate correlations between each feature and the target
- If any feature has correlation > 0.95, investigate whether it's legitimate or leakage
- For time series: Be especially careful with rolling windows, lag features, or any transformation of the target variable

**Example of Problematic Feature:**
- `air_temp_rolling_7h` (7-hour rolling mean of Air Temperature) when predicting Air Temperature
- This feature has ~99.4% correlation with the target - too high to be useful and indicates circular logic

**Solution:**
- Only create rolling windows for **predictor variables**, not the target
- Use rolling windows of: Wind Speed, Humidity, Barometric Pressure, etc.
- Avoid rolling windows of: Air Temperature (if that's your target)

---

## Required Artifacts

You must create exactly these 3 files in the `output/` directory:

### 1. `output/q7_predictions.csv`
**Format:** CSV file
**Required Columns (exact names):**
- `actual` - Actual target values from test set
- `predicted_linear` or `predicted_model1` - Predictions from first model (e.g., Linear Regression)
- `predicted_xgboost` or `predicted_model2` - Predictions from second model (e.g., XGBoost)
- Additional columns for additional models (e.g., `predicted_random_forest` or `predicted_model3`)

**Requirements:**
- Must have at least 2 model prediction columns (in addition to `actual`)
- All values must be numeric (float)
- Same number of rows as test set
- **No index column** (save with `index=False`)

**Example:**
```csv
actual,predicted_linear,predicted_xgboost
15.2,14.8,15.1
15.3,15.0,15.2
...
```

### 2. `output/q7_model_metrics.txt`
**Format:** Plain text file
**Content:** Performance metrics for each model
**Required information for each model:**
- Model name
- At least one metric for both train and test sets
  - **Suggested:** R² score (works for both Linear Regression and Random Forest)
  - **Alternatives:** RMSE or MAE (also work for both models)
- Additional metrics are optional but recommended

**Requirements:**
- Clearly labeled (model name, metric name)
- At minimum: one metric (e.g., R²) for train and test for each model
- Format should be readable

**Example format (minimum required - one metric, R² shown):**
```
MODEL PERFORMANCE METRICS
========================

LINEAR REGRESSION:
  Train R²: 0.3048
  Test R²:  0.3046

XGBOOST:
  Train R²: 0.9091
  Test R²:  0.7684
```

**Note:** R² (R-squared) works for both Linear Regression and XGBoost. It's a universal regression metric that measures the proportion of variance explained. Alternative metrics like RMSE or MAE also work well for both models.

**Example format (with additional metrics - recommended):**
```
MODEL PERFORMANCE METRICS
========================

LINEAR REGRESSION:
  Train R²: 0.3048
  Test R²:  0.3046
  Train RMSE: 8.42
  Test RMSE:  8.43
  Train MAE:  7.03
  Test MAE:   7.04

XGBOOST:
  Train R²: 0.9091
  Test R²:  0.7684
  Train RMSE: 3.45
  Test RMSE:  4.87
  Train MAE:  2.58
  Test MAE:   3.66
```

### 3. `output/q7_feature_importance.csv`
**Format:** CSV file
**Required Columns (exact names):** `feature`, `importance`
**Content:** Feature importance from tree-based models (XGBoost, Random Forest)
**Requirements:**
- One row per feature
- `feature`: Feature name (string)
- `importance`: Importance score (float, typically 0-1, sum to 1)
- Sorted by importance (descending)
- **No index column** (save with `index=False`)

**Note:** If using Linear Regression (which doesn't have feature importance), you can skip this file or use coefficient magnitudes as importance. However, tree-based models like XGBoost are preferred as they provide feature importance.

**Example:**
```csv
feature,importance
Air Temperature,0.6539
hour,0.1234
month,0.0892
Water Temperature,0.0456
...
```

---

## Requirements Checklist

- [ ] At least 2 different models trained
  - **Suggested:** Linear Regression and XGBoost (these work well and demonstrate different modeling approaches)
  - You may choose other models if appropriate (e.g., Random Forest, Gradient Boosting, etc.)
- [ ] Performance evaluated on both train and test sets
- [ ] Models compared
- [ ] Feature importance extracted (if applicable - tree-based models)
- [ ] Model performance documented with at least one metric
  - **Suggested:** R² score (works for both Linear Regression and XGBoost, and all regression models)
  - Alternative metrics that work for both: RMSE or MAE
  - You may include additional metrics if relevant (e.g., MAPE, adjusted R²)
- [ ] All 3 required artifacts saved with exact filenames

---

## Your Approach

1. **Train Model 1 (e.g., Linear Regression):**
   ```python
   lr_model = LinearRegression()
   lr_model.fit(X_train, y_train)
   lr_train_pred = lr_model.predict(X_train)
   lr_test_pred = lr_model.predict(X_test)
   ```

2. **Train Model 2 (e.g., XGBoost):**
   ```python
   xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42)
   xgb_model.fit(X_train, y_train)
   xgb_train_pred = xgb_model.predict(X_train)
   xgb_test_pred = xgb_model.predict(X_test)
   ```

3. **Train Model 3 (optional, e.g., Random Forest):**
   ```python
   rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
   rf_model.fit(X_train, y_train)
   rf_train_pred = rf_model.predict(X_train)
   rf_test_pred = rf_model.predict(X_test)
   ```

4. **Calculate metrics:**
   ```python
   # At minimum, calculate one metric for each model
   # R² works for both Linear Regression and XGBoost (suggested)
   train_r2 = r2_score(y_train, train_pred)
   test_r2 = r2_score(y_test, test_pred)
   
   # Alternative metrics that also work for both models:
   # RMSE (Root Mean Squared Error)
   train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
   test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
   
   # MAE (Mean Absolute Error)
   train_mae = mean_absolute_error(y_train, train_pred)
   test_mae = mean_absolute_error(y_test, test_pred)
   ```

5. **Extract feature importance (tree-based models):**
   ```python
   feature_importance = pd.DataFrame({
       'feature': X_train.columns,
       'importance': xgb_model.feature_importances_
   }).sort_values('importance', ascending=False)
   feature_importance.to_csv('output/q7_feature_importance.csv', index=False)
   ```

6. **Save predictions:**
   ```python
   predictions_df = pd.DataFrame({
       'actual': y_test.values,
       'predicted_linear': lr_test_pred,
       'predicted_xgboost': xgb_test_pred
   })
   predictions_df.to_csv('output/q7_predictions.csv', index=False)
   ```

7. **Save metrics:**
   - Write metrics to `output/q7_model_metrics.txt`

---

## Decision Points

- **Model selection:** Train at least 2 different models. We suggest starting with **Linear Regression** and **XGBoost** - these work well and demonstrate different modeling approaches (linear vs gradient boosting). You may choose other models if appropriate (e.g., Random Forest, Gradient Boosting, etc.). See Lecture 11 Notebook 4 for examples.
- **Evaluation metrics:** Report at least one metric for each model. We suggest **R² score** (coefficient of determination) - it works for both Linear Regression and XGBoost, and all regression models. It measures the proportion of variance explained and is easy to interpret. Alternative metrics that work well for both models include **RMSE** (Root Mean Squared Error) or **MAE** (Mean Absolute Error). You may include additional metrics if relevant (e.g., MAPE, adjusted R²). Compare train vs test performance to check for overfitting.
- **Feature importance:** If using tree-based models (like XGBoost), extract feature importance to understand which features matter most.

---

## Interpreting Model Performance

**Warning Signs of Data Leakage:**
- R² = 1.0000 (perfect score) or R² > 0.999
- RMSE or MAE = 0.0 or unrealistically small (< 0.01 for temperature)
- Train and test performance nearly identical (difference < 0.01)
- Any feature with correlation > 0.99 with target

**Realistic Expectations:**
- For temperature prediction: RMSE of 0.5-2.0°C is realistic
- R² of 0.85-0.98 is strong but realistic
- Some difference between train and test performance is normal

**If you see warning signs:**
1. Check your features for data leakage (see Data Leakage Warning above)
2. Calculate correlations between features and target
3. Remove features that are transformations of the target variable
4. Re-train models and verify performance is now realistic

---

## Checkpoint

After Q7, you should have:
- [ ] At least 2 models trained (suggested: Linear Regression and XGBoost)
- [ ] Performance metrics calculated (at minimum: one metric like R², RMSE, or MAE for train and test; additional metrics recommended)
- [ ] Models compared
- [ ] Feature importance extracted (if applicable - tree-based models like XGBoost)
- [ ] All 3 artifacts saved: `q7_predictions.csv`, `q7_model_metrics.txt`, `q7_feature_importance.csv`

---

**Next:** Continue to `q8_results.md` for Results.

