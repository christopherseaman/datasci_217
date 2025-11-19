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

**Note:** This question focuses on statistical modeling for inference - understanding relationships between variables. We'll use a subset of features (`MedInc`, `AveBedrms`, `Population`) to focus on interpretability and statistical significance rather than maximizing prediction accuracy. The `statsmodels` library provides detailed statistical information (p-values, confidence intervals, AIC) that helps us understand *why* variables are related.

**Why a subset of features?** In statistical modeling, we often use fewer features to maintain interpretability and focus on understanding relationships. This contrasts with machine learning (Question 2), where we use all available features to maximize prediction accuracy.

**Objective:** Fit linear regression models using `statsmodels`, extract statistical information, and compare models with and without interaction terms.

### Part 1.1: Fit the Model

Fit a linear regression model predicting `house_value` from `MedInc`, `AveBedrms`, and `Population` using the formula API.

```python
# TODO: Fit a linear regression model using statsmodels formula API
# Hint: Use smf.ols() with the formula 'house_value ~ MedInc + AveBedrms + Population'
# Don't forget to call .fit() on the model

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

# TODO: Extract p-values for coefficients
# Use: results.pvalues to get p-values for each coefficient
# Print which coefficients are statistically significant (p < 0.05)

pvalues = None  # Replace None with your code
print("\n=== Coefficient Significance (p-values) ===")
# Your code here to print p-values and identify significant coefficients

# TODO: Save key statistics to output file
# Extract: R-squared, number of observations, and AIC (Akaike Information Criterion)
# Format: "R-squared: X.XXXX\nObservations: XXXX\nAIC: XXXXX.XX"
#
# Example output:
# R-squared: 0.4743
# Observations: 20640
# AIC: 51221.28

with open('output/q1_model_summary.txt', 'w') as f:
    # Your code here
    pass
```

### Part 1.3: Make Predictions

Make predictions for all houses and save to CSV.

**Note:** In statistical modeling, we often make predictions on the full dataset to understand model fit. This differs from machine learning (Question 2), where we use train/test splits to evaluate generalization.

```python
# TODO: Make predictions using the fitted model
# Hint: Use results.predict() with a DataFrame containing the features used in the model
# The features are: MedInc, AveBedrms, Population
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

### Part 1.4: Model with Interaction Term

Now let's fit a model with an interaction term. An **interaction term** allows the effect of one variable to depend on the value of another variable. For example, the effect of income (`MedInc`) on house value might depend on the number of bedrooms (`AveBedrms`). In the formula API, we use `*` to include both main effects and their interaction.

```python
# TODO: Fit a model with an interaction term between MedInc and AveBedrms
# Hint: Use formula 'house_value ~ MedInc + AveBedrms + Population + MedInc:AveBedrms'
# Or use 'house_value ~ MedInc * AveBedrms + Population' (the * includes both main effects and interaction)

model_interaction = None  # Replace None with your code
results_interaction = None  # Replace None with your code

print("\n=== Model with Interaction Term ===")
# Your code here to print the summary
```

### Part 1.5: Compare Models

Compare the two models using AIC (Akaike Information Criterion). Lower AIC indicates a better model (accounting for model complexity).

```python
# TODO: Compare the two models using AIC
# Extract AIC from both models: results.aic and results_interaction.aic
# Determine which model is better (lower AIC is better)

aic_simple = None  # Replace None with your code
aic_interaction = None  # Replace None with your code

print("\n=== Model Comparison ===")
print(f"Simple model AIC: {aic_simple:.2f}")
print(f"Interaction model AIC: {aic_interaction:.2f}")
# Your code here to determine and print which model is better
```

---

## Question 2: Machine Learning with scikit-learn

**Note:** While Question 1 focused on statistical inference (understanding relationships and testing hypotheses), Question 2 focuses on machine learning for prediction. We'll use all available features to maximize prediction accuracy rather than focusing on interpretability.

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

Fit a linear regression model and evaluate it on both training and test sets. Comparing train and test performance helps us detect overfitting - if the model performs much better on training data than test data, it's likely overfitting.

```python
# TODO: Fit a LinearRegression model
lr_model = None  # Replace None with your code
# Your code here

# TODO: Make predictions on both training and test sets
lr_train_pred = None  # Replace None with your code
lr_test_pred = None  # Replace None with your code

# Calculate metrics on both sets
lr_train_r2 = r2_score(y_train, lr_train_pred)
lr_test_r2 = r2_score(y_test, lr_test_pred)
lr_train_rmse = np.sqrt(mean_squared_error(y_train, lr_train_pred))
lr_test_rmse = np.sqrt(mean_squared_error(y_test, lr_test_pred))

print("=== Linear Regression Results ===")
print(f"Training - R²: {lr_train_r2:.4f}, RMSE: {lr_train_rmse:.2f}")
print(f"Test - R²: {lr_test_r2:.4f}, RMSE: {lr_test_rmse:.2f}")

# Store test predictions for later use
lr_pred = lr_test_pred
lr_r2 = lr_test_r2
lr_rmse = lr_test_rmse
```

### Part 2.3: Fit Random Forest

Fit a random forest model and evaluate it on both training and test sets.

```python
# TODO: Fit a RandomForestRegressor model
# Use: n_estimators=50, max_depth=8, random_state=42
rf_model = None  # Replace None with your code
# Your code here

# TODO: Make predictions on both training and test sets
rf_train_pred = None  # Replace None with your code
rf_test_pred = None  # Replace None with your code

# Calculate metrics on both sets
rf_train_r2 = r2_score(y_train, rf_train_pred)
rf_test_r2 = r2_score(y_test, rf_test_pred)
rf_train_rmse = np.sqrt(mean_squared_error(y_train, rf_train_pred))
rf_test_rmse = np.sqrt(mean_squared_error(y_test, rf_test_pred))

print("=== Random Forest Results ===")
print(f"Training - R²: {rf_train_r2:.4f}, RMSE: {rf_train_rmse:.2f}")
print(f"Test - R²: {rf_test_r2:.4f}, RMSE: {rf_test_rmse:.2f}")

# Store test predictions and metrics for later use
rf_pred = rf_test_pred
rf_r2 = rf_test_r2
rf_rmse = rf_test_rmse

# TODO: Extract feature importance for later comparison
# Use: rf_model.feature_importances_
rf_feature_importance = None  # Replace None with your code
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
# Include both train and test metrics
# Format: "Linear Regression - Train R²: X.XXXX, Test R²: X.XXXX, Test RMSE: XX.XX\nRandom Forest - Train R²: X.XXXX, Test R²: X.XXXX, Test RMSE: XX.XX"
#
# Example output:
# Linear Regression - Train R²: 0.6126, Test R²: 0.5758, Test RMSE: 0.75
# Random Forest - Train R²: 0.8050, Test R²: 0.7389, Test RMSE: 0.58

with open('output/q2_model_comparison.txt', 'w') as f:
    # Your code here
    pass
```

---

## Question 3: Gradient Boosting with XGBoost

**Note:** Question 3 introduces gradient boosting, an advanced machine learning technique that often achieves the best performance on tabular data. XGBoost builds models sequentially, with each new model learning from the mistakes of previous ones.

**Objective:** Fit an XGBoost model and extract feature importance.

### Part 3.1: Fit XGBoost Model

Fit an XGBoost regressor model.

```python
# TODO: Fit an XGBRegressor model
# Use: n_estimators=100, max_depth=3, learning_rate=0.15, random_state=42
xgb_model = None  # Replace None with your code
# Your code here

# TODO: Make predictions on test set
xgb_pred = None  # Replace None with your code

# Calculate metrics
xgb_r2 = r2_score(y_test, xgb_pred)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))

print(f"XGBoost - R²: {xgb_r2:.4f}, RMSE: {xgb_rmse:.2f}")
```

### Part 3.2: Extract and Compare Feature Importance

Extract feature importance from XGBoost and compare it with Random Forest from Question 2.

```python
# TODO: Extract feature importance from XGBoost
# Use: xgb_model.feature_importances_
xgb_feature_importance = None  # Replace None with your code

# Create DataFrame for XGBoost importance
xgb_importance_df = pd.DataFrame({
    'feature': feature_cols,
    'xgb_importance': xgb_feature_importance
}).sort_values('xgb_importance', ascending=False)

print("\n=== XGBoost Feature Importance ===")
print(xgb_importance_df)

# TODO: Compare with Random Forest feature importance
# Create a comparison DataFrame with both models' feature importance
# Sort by XGBoost importance for display

importance_comparison = pd.DataFrame({
    'feature': feature_cols,
    'random_forest': rf_feature_importance,
    'xgboost': xgb_feature_importance
}).sort_values('xgboost', ascending=False)

print("\n=== Feature Importance Comparison ===")
print(importance_comparison)

# TODO: Save XGBoost feature importance to text file
# Format: "feature_name: X.XXXX" (one per line, sorted by importance)
#
# Example output (first few lines):
# MedInc: 0.5677
# AveOccup: 0.1542
# Longitude: 0.0743

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

