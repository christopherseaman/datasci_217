# Assignment 10: Modeling Fundamentals

Complete the following three questions to demonstrate your understanding of statistical modeling, machine learning, and gradient boosting.

## Setup

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import os

# Create output directory
os.makedirs('output', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)
```

## Load Data

```python
# Load California Housing dataset from scikit-learn
from sklearn.datasets import fetch_california_housing

# Fetch the dataset
housing_data = fetch_california_housing(as_frame=True)
df = housing_data.frame

# Rename target for clarity
df = df.rename(columns={'MedHouseVal': 'house_value'})

print(f"Loaded {len(df)} housing records")
print("\nFeature names:", housing_data.feature_names)
print("\nFirst few rows:")
print(df.head())
print("\nSummary statistics:")
print(df.describe())
```

---

## Question 1: Statistical Modeling with statsmodels

**Objective:** Fit a linear regression model using `statsmodels` and extract statistical information.

### Part 1.1: Fit the Model

Fit a linear regression model predicting `house_value` from `MedInc`, `HouseAge`, and `AveRooms` using the formula API.

```python
# TODO: Fit a linear regression model using statsmodels formula API
# Use: smf.ols('house_value ~ MedInc + HouseAge + AveRooms', data=df)
# Hint: Don't forget to call .fit() on the model

model = None  # Replace None with your code
results = None  # Replace None with your code
```

### Part 1.2: Extract Model Summary

Print the model summary and save key statistics to a text file.

```python
# TODO: Print the model summary
# Use: results.summary()

print("=== Model Summary ===")
# Your code here

# TODO: Save key statistics to output file
# Extract: R-squared, number of observations, and F-statistic p-value
# Format: "R-squared: X.XXXX\nObservations: XXXX\nF-statistic p-value: X.XXe-XX"

with open('output/q1_model_summary.txt', 'w') as f:
    # Your code here
    pass
```

### Part 1.3: Make Predictions

Make predictions for all houses and save to CSV.

```python
# TODO: Make predictions using the fitted model
# Use: results.predict(df[['MedInc', 'HouseAge', 'AveRooms']])
# Save predictions along with actual values to CSV

predictions = None  # Replace None with your code

# Create DataFrame with predictions
pred_df = pd.DataFrame({
    'actual_value': df['house_value'],
    'predicted_value': predictions
})

# Save to CSV
pred_df.to_csv('output/q1_statistical_model.csv', index=False)
print(f"\nSaved {len(pred_df)} predictions to output/q1_statistical_model.csv")
```

---

## Question 2: Machine Learning with scikit-learn

**Objective:** Fit and compare linear regression and random forest models using `scikit-learn`.

### Part 2.1: Prepare Data

Split the data into training and test sets.

```python
# TODO: Prepare features and target
# Features: ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
# Target: 'house_value'

feature_cols = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
X = None  # Replace None with your code
y = None  # Replace None with your code

# TODO: Split into train and test sets (80/20 split, random_state=42)
X_train, X_test, y_train, y_test = None  # Replace None with your code

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
```

### Part 2.2: Fit Linear Regression

Fit a linear regression model and evaluate it.

```python
# TODO: Fit a LinearRegression model
lr_model = None  # Replace None with your code
# Your code here

# TODO: Make predictions on test set
lr_pred = None  # Replace None with your code

# Calculate metrics
lr_r2 = r2_score(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))

print(f"Linear Regression - R²: {lr_r2:.4f}, RMSE: {lr_rmse:.2f}")
```

### Part 2.3: Fit Random Forest

Fit a random forest model and evaluate it.

```python
# TODO: Fit a RandomForestRegressor model
# Use: n_estimators=100, random_state=42
rf_model = None  # Replace None with your code
# Your code here

# TODO: Make predictions on test set
rf_pred = None  # Replace None with your code

# Calculate metrics
rf_r2 = r2_score(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

print(f"Random Forest - R²: {rf_r2:.4f}, RMSE: {rf_rmse:.2f}")
```

### Part 2.4: Save Predictions and Comparison

Save predictions and model comparison to files.

```python
# TODO: Save predictions to CSV
# Include: actual_value, lr_predicted_value, rf_predicted_value

pred_df = pd.DataFrame({
    'actual_value': y_test.values,
    'lr_predicted_value': lr_pred,
    'rf_predicted_value': rf_pred
})

pred_df.to_csv('output/q2_ml_predictions.csv', index=False)
print(f"\nSaved predictions to output/q2_ml_predictions.csv")

# TODO: Save model comparison to text file
# Format: "Linear Regression - R²: X.XXXX, RMSE: XX.XX\nRandom Forest - R²: X.XXXX, RMSE: XX.XX"

with open('output/q2_model_comparison.txt', 'w') as f:
    # Your code here
    pass
```

---

## Question 3: Gradient Boosting with XGBoost

**Objective:** Fit an XGBoost model and extract feature importance.

### Part 3.1: Fit XGBoost Model

Fit an XGBoost regressor model.

```python
# TODO: Fit an XGBRegressor model
# Use: n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
xgb_model = None  # Replace None with your code
# Your code here

# TODO: Make predictions on test set
xgb_pred = None  # Replace None with your code

# Calculate metrics
xgb_r2 = r2_score(y_test, xgb_pred)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))

print(f"XGBoost - R²: {xgb_r2:.4f}, RMSE: {xgb_rmse:.2f}")
```

### Part 3.2: Extract Feature Importance

Extract and save feature importance.

```python
# TODO: Extract feature importance
# Use: xgb_model.feature_importances_
feature_importance = None  # Replace None with your code

# Create DataFrame
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(importance_df)

# TODO: Save feature importance to text file
# Format: "feature_name: X.XXXX" (one per line, sorted by importance)

with open('output/q3_feature_importance.txt', 'w') as f:
    # Your code here
    pass
```

### Part 3.3: Save Predictions

Save XGBoost predictions to CSV.

```python
# TODO: Save predictions to CSV
# Include: actual_value, xgb_predicted_value

pred_df = pd.DataFrame({
    'actual_value': y_test.values,
    'xgb_predicted_value': xgb_pred
})

pred_df.to_csv('output/q3_xgboost_model.csv', index=False)
print(f"\nSaved predictions to output/q3_xgboost_model.csv")
```

---

## Submission Checklist

Before submitting, verify you've created all required output files:

- [ ] `output/q1_statistical_model.csv`
- [ ] `output/q1_model_summary.txt`
- [ ] `output/q2_ml_predictions.csv`
- [ ] `output/q2_model_comparison.txt`
- [ ] `output/q3_xgboost_model.csv`
- [ ] `output/q3_feature_importance.txt`

Run the tests locally to verify:

```bash
pytest -q 10/assignment/.github/test/test_assignment.py
```

