# Q7: Modeling

**Phase 8:** Modeling  
**Points: 9 points**

**Focus:** Train multiple models, evaluate performance, compare models, extract feature importance.

**Lecture Reference:** Lecture 11, Notebook 4 ([`11/demo/04_modeling_results.ipynb`](https://github.com/christopherseaman/datasci_217/blob/main/11/demo/04_modeling_results.ipynb)), Phase 8. Also see Lecture 10 (modeling with sklearn and XGBoost).

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
- At least R² score for both train and test sets (additional metrics like RMSE, MAE recommended but optional)

**Requirements:**
- Clearly labeled (model name, metric name)
- **At minimum:** R² (or R-squared or R^2) for train and test for each model
- Additional metrics (RMSE, MAE) are recommended for a complete analysis
- Format should be readable

**Example format (minimum - R² only):**
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

**Example format (recommended - with additional metrics):**
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

**Note:** Tree-based models (XGBoost, Random Forest) provide feature importance directly via `.feature_importances_`. If using only Linear Regression, you can use the absolute values of coefficients as a proxy for importance.

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
  - **Suggested:** Linear Regression and XGBoost (or Random Forest)
  - You may choose other models if appropriate
- [ ] Performance evaluated on both train and test sets
- [ ] Models compared
- [ ] Feature importance extracted
  - Tree-based models: use `.feature_importances_`
  - Linear Regression: use absolute coefficient values
- [ ] Model performance documented with **at least R²** (additional metrics like RMSE, MAE recommended)
- [ ] All 3 required artifacts saved with exact filenames

---

## Your Approach

1. **Check for data leakage** - Before training, compute correlations between features and target. Any feature with correlation > 0.95 should be investigated and considered for removal.
2. **Train at least 2 models** - Fit models to training data, generate predictions for both train and test sets
3. **Calculate metrics** - At minimum R² for train and test; RMSE and MAE recommended
4. **Extract feature importance** - Use `.feature_importances_` for tree-based models, or coefficient magnitudes for linear models
5. **Save predictions** - DataFrame with `actual` column plus `predicted_*` columns for each model
6. **Save metrics** - Write clearly labeled metrics to text file

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

