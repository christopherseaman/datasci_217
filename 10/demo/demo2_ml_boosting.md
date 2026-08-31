# Demo 2: Machine Learning with scikit-learn and XGBoost

## Learning Objectives
- Master the scikit-learn fit/predict pattern
- Build and evaluate linear regression and random forest models
- Use XGBoost for gradient boosting
- Distinguish model-specific importance from held-out permutation importance
- Compare model performance
- Visualize results with Altair

## Setup

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
import xgboost as xgb
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

## Part 2: Train/Validation/Test Split

The golden rule: never evaluate on data the model has seen during training!

Before we can train any machine learning model, we need to split our data. The validation set selects models and tuning choices; the test set stays untouched until one final evaluation of the frozen choice.

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

**Why split the data?**
- **Training set**: Used to teach the model patterns in the data
- **Validation set**: Used to compare candidate models and tune training choices
- **Test set**: Held untouched until the final, one-time evaluation
- **60/20/20 split**: A simple teaching split; proportions depend on dataset size
- **random_state=42**: Ensures reproducible splits (same random seed = same split)

## Part 3: Linear Regression with scikit-learn

scikit-learn's API is consistent across all models: create, fit, predict.

The scikit-learn workflow is beautifully simple: create the model, fit it to training data, then make predictions. This same pattern works for almost every model in scikit-learn.

```python
# Create and fit linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions
y_train_pred = lr_model.predict(X_train)
y_valid_pred = lr_model.predict(X_valid)

# Evaluate model
train_r2 = r2_score(y_train, y_train_pred)
valid_r2 = r2_score(y_valid, y_valid_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
valid_rmse = np.sqrt(mean_squared_error(y_valid, y_valid_pred))

print("=== Linear Regression Results ===")
print(f"Training R²: {train_r2:.4f}")
print(f"Validation R²: {valid_r2:.4f}")
print(f"Training RMSE: ${train_rmse:.2f}k")
print(f"Validation RMSE: ${valid_rmse:.2f}k")
print(f"\nCoefficients:")
coef_df = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': lr_model.coef_
})
print(coef_df)
print(f"\nIntercept: ${lr_model.intercept_:.2f}k")
```

**Understanding the metrics:**
- **R² (R-squared)**: Proportion of variance explained (0-1, higher is better). An R² of 0.85 means the model explains 85% of house value variation.
- **RMSE (Root Mean Squared Error)**: Average prediction error in the same units as the target. Lower is better.
- **Training vs validation**: If training performance is much better than validation, the model may be overfitting.
- **Coefficients**: Show how much each feature contributes to the house value prediction.

## Part 4: Regularized Linear Models

Regularization helps prevent overfitting by penalizing large coefficients.

Regularization is a technique to prevent overfitting by penalizing large coefficients. Think of it as adding a "simplicity penalty" - the model is rewarded for using smaller coefficients.

**Ridge (L2) regularization** shrinks all coefficients toward zero but doesn't eliminate them. **Lasso (L1) regularization** can completely zero out coefficients, effectively performing automatic feature selection.

```python
# Ridge Regression (L2 regularization)
ridge_model = Ridge(alpha=10.0)  # alpha controls regularization strength
ridge_model.fit(X_train, y_train)
ridge_valid_r2 = r2_score(y_valid, ridge_model.predict(X_valid))
ridge_valid_rmse = np.sqrt(mean_squared_error(y_valid, ridge_model.predict(X_valid)))

# Lasso Regression (L1 regularization - can zero out coefficients)
lasso_model = Lasso(alpha=1.0)
lasso_model.fit(X_train, y_train)
lasso_valid_r2 = r2_score(y_valid, lasso_model.predict(X_valid))
lasso_valid_rmse = np.sqrt(mean_squared_error(y_valid, lasso_model.predict(X_valid)))

print("=== Regularized Models Comparison ===")
print(f"Linear Regression - Validation R²: {valid_r2:.4f}, RMSE: ${valid_rmse:.2f}k")
print(f"Ridge Regression - Validation R²: {ridge_valid_r2:.4f}, RMSE: ${ridge_valid_rmse:.2f}k")
print(f"Lasso Regression - Validation R²: {lasso_valid_r2:.4f}, RMSE: ${lasso_valid_rmse:.2f}k")

# Compare coefficients
coef_comparison = pd.DataFrame({
    'feature': feature_cols,
    'linear': lr_model.coef_,
    'ridge': ridge_model.coef_,
    'lasso': lasso_model.coef_
})
print("\n=== Coefficient Comparison ===")
print(coef_comparison)

# Lasso can zero out features (feature selection)
print(f"\nFeatures selected by Lasso (non-zero coefficients): {sum(lasso_model.coef_ != 0)}")
```

**When to use regularization:**
- **Many features**: Regularization helps when you have more features than observations
- **Multicollinearity**: When features are highly correlated, regularization stabilizes estimates
- **Feature selection**: Lasso automatically identifies the most important features
- **Overfitting prevention**: Both methods help models generalize better to new data

## Part 5: Random Forest

Random Forest is an ensemble method that handles non-linear relationships automatically.

```python
# Create and fit Random Forest
rf_model = RandomForestRegressor(
    n_estimators=100,  # Number of trees
    max_depth=10,      # Maximum tree depth
    min_samples_split=5,  # Minimum samples to split
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)
rf_model.fit(X_train, y_train)

# Make predictions
rf_train_pred = rf_model.predict(X_train)
rf_valid_pred = rf_model.predict(X_valid)

# Evaluate
rf_train_r2 = r2_score(y_train, rf_train_pred)
rf_valid_r2 = r2_score(y_valid, rf_valid_pred)
rf_train_rmse = np.sqrt(mean_squared_error(y_train, rf_train_pred))
rf_valid_rmse = np.sqrt(mean_squared_error(y_valid, rf_valid_pred))

print("=== Random Forest Results ===")
print(f"Training R²: {rf_train_r2:.4f}")
print(f"Validation R²: {rf_valid_r2:.4f}")
print(f"Training RMSE: ${rf_train_rmse:.2f}k")
print(f"Validation RMSE: ${rf_valid_rmse:.2f}k")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== Feature Importance ===")
print(feature_importance)
```

**Key insights:**
- Random Forest often outperforms linear models on complex, non-linear data
- Its model-specific importance summarizes how the fitted forest used the features
- Random Forest can capture interactions between features automatically

## Part 6: XGBoost - The Secret Weapon

XGBoost is a powerful gradient boosting library that often wins competitions.

```python
# Create and fit XGBoost model
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=False
)

# Make predictions
xgb_train_pred = xgb_model.predict(X_train)
xgb_valid_pred = xgb_model.predict(X_valid)

# Evaluate
xgb_train_r2 = r2_score(y_train, xgb_train_pred)
xgb_valid_r2 = r2_score(y_valid, xgb_valid_pred)
xgb_train_rmse = np.sqrt(mean_squared_error(y_train, xgb_train_pred))
xgb_valid_rmse = np.sqrt(mean_squared_error(y_valid, xgb_valid_pred))

print("=== XGBoost Results ===")
print(f"Training R²: {xgb_train_r2:.4f}")
print(f"Validation R²: {xgb_valid_r2:.4f}")
print(f"Training RMSE: ${xgb_train_rmse:.2f}k")
print(f"Validation RMSE: ${xgb_valid_rmse:.2f}k")

# Feature importance
xgb_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== XGBoost Feature Importance ===")
print(xgb_importance)
```

## Part 7: Model Comparison

Let's compare all our models side-by-side.

```python
# Compare all models
comparison = pd.DataFrame({
    'Model': ['Linear Regression', 'Ridge', 'Lasso', 'Random Forest', 'XGBoost'],
    'Train R²': [train_r2, r2_score(y_train, ridge_model.predict(X_train)), 
                  r2_score(y_train, lasso_model.predict(X_train)), rf_train_r2, xgb_train_r2],
    'Validation R²': [valid_r2, ridge_valid_r2, lasso_valid_r2, rf_valid_r2, xgb_valid_r2],
    'Validation RMSE': [valid_rmse, ridge_valid_rmse, lasso_valid_rmse, rf_valid_rmse, xgb_valid_rmse]
})

print("=== Model Comparison ===")
print(comparison.to_string(index=False))

# Visualize comparison
comparison_long = comparison.melt(
    id_vars='Model',
    value_vars=['Train R²', 'Validation R²'],
    var_name='Metric',
    value_name='R² Score'
)

alt.Chart(comparison_long).mark_bar().encode(
    x=alt.X('Model:N', title='Model', sort='-y'),
    y=alt.Y('R² Score:Q', title='R² Score', scale=alt.Scale(domain=[0, 1])),
    color='Metric:N',
    column='Metric:N'
).properties(
    width=150,
    height=300
)
```

## Part 8: Validation Prediction Visualization

Visualize how well our best model predicts house values.

```python
# Use validation predictions while the test set remains sealed.
pred_df = pd.DataFrame({
    'actual': y_valid.values,
    'predicted': xgb_valid_pred,
    'error': y_valid.values - xgb_valid_pred
})

# Scatter plot: actual vs predicted
scatter = alt.Chart(pred_df).mark_circle(opacity=0.5).encode(
    x=alt.X('actual:Q', title='Actual House Value ($k)'),
    y=alt.Y('predicted:Q', title='Predicted House Value ($k)'),
    color=alt.Color('error:Q', scale=alt.Scale(scheme='redblue', domainMid=0), 
                    title='Error ($k)')
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
    x=alt.X('predicted:Q', title='Predicted House Value ($k)'),
    y=alt.Y('error:Q', title='Residual (Actual - Predicted)'),
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

## Part 9: Feature Importance Comparison

Compare feature importance across tree-based models.

```python
# Combine feature importance from both models
importance_comparison = pd.merge(
    feature_importance.rename(columns={'importance': 'random_forest'}),
    xgb_importance.rename(columns={'importance': 'xgboost'}),
    on='feature'
)

importance_long = importance_comparison.melt(
    id_vars='feature',
    value_vars=['random_forest', 'xgboost'],
    var_name='model',
    value_name='importance'
)

alt.Chart(importance_long).mark_bar().encode(
    x=alt.X('importance:Q', title='Feature Importance'),
    y=alt.Y('feature:N', title='Feature', sort='-x'),
    color='model:N',
    column='model:N'
).properties(
    width=200,
    height=300
)
```

The summaries above are specific to each tree implementation. Permutation
importance works with any fitted predictor and measures how much its held-out
score deteriorates when one feature is shuffled. Here we use the validation set,
not the final test set, and keep MAE as the evaluation measure. scikit-learn
orients scorers so higher is better, so MAE uses the negative-MAE scorer; a
positive result below is the increase in validation MAE after shuffling.

```python
# Measure predictive reliance on validation data with reproducible shuffles.
permutation_result = permutation_importance(
    xgb_model,
    X_valid,
    y_valid,
    scoring='neg_mean_absolute_error',
    n_repeats=10,
    random_state=42,
    n_jobs=-1,
)

permutation_df = pd.DataFrame({
    'feature': feature_cols,
    'validation_mae_increase': permutation_result.importances_mean,
    'repeat_std': permutation_result.importances_std,
}).sort_values('validation_mae_increase', ascending=False)

print("=== XGBoost Permutation Importance (Validation MAE Increase) ===")
print(permutation_df)
```

Correlated features can substitute for one another, so shuffling either one may
show little damage or divide importance between them. These values describe this
fitted model's predictive reliance under shuffling; they are not causal effects
and do not establish that changing a feature would change house values.

## Part 10: Early Stopping with XGBoost

Early stopping prevents overfitting by stopping training when validation performance stops improving.

```python
# XGBoost with early stopping
# Note: In XGBoost 2.0+, early_stopping_rounds is passed to the constructor
xgb_early_stop = xgb.XGBRegressor(
    n_estimators=500,  # Set high, but early stopping will stop earlier
    max_depth=5,
    learning_rate=0.1,
    early_stopping_rounds=10,  # Stop if no improvement for 10 rounds
    random_state=42,
    n_jobs=-1
)

# Fit with early stopping
xgb_early_stop.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=False
)

# Check how many rounds were actually used
print(f"=== Early Stopping Results ===")
print(f"Best iteration: {xgb_early_stop.best_iteration}")
print(f"Best score: {xgb_early_stop.best_score:.4f}")

# Compare with model without early stopping on validation data.
xgb_early_pred = xgb_early_stop.predict(X_valid)
xgb_early_r2 = r2_score(y_valid, xgb_early_pred)
xgb_early_rmse = np.sqrt(mean_squared_error(y_valid, xgb_early_pred))

print(f"\nXGBoost (no early stopping) - Validation R²: {xgb_valid_r2:.4f}, RMSE: ${xgb_valid_rmse:.2f}k")
print(f"XGBoost (with early stopping) - Validation R²: {xgb_early_r2:.4f}, RMSE: ${xgb_early_rmse:.2f}k")
```

## Part 11: One Final Test Evaluation

After selecting early-stopped XGBoost using validation results, refit that frozen
configuration on the combined training and validation rows. This is the first and
only point at which the test set is used.

```python
final_xgb = xgb.XGBRegressor(
    n_estimators=xgb_early_stop.best_iteration + 1,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
)
final_xgb.fit(X_train_valid, y_train_valid, verbose=False)
final_test_pred = final_xgb.predict(X_test)
final_test_r2 = r2_score(y_test, final_test_pred)
final_test_rmse = np.sqrt(mean_squared_error(y_test, final_test_pred))

print("=== Final Test Performance: selected early-stopped XGBoost ===")
print(f"Test R²: {final_test_r2:.4f}")
print(f"Test RMSE: ${final_test_rmse:.2f}k")
```

## Key Takeaways

1. **scikit-learn API**: Consistent fit/predict pattern across all models
2. **Train/validation/test split**: Select with validation; report the test result once
3. **Regularization**: Ridge and Lasso help prevent overfitting
4. **Random Forest**: Handles non-linear relationships automatically
5. **XGBoost**: A strong candidate to benchmark on tabular data
6. **Feature importance**: Compare model-specific summaries with held-out permutation importance
7. **Early stopping**: Prevents overfitting in gradient boosting
8. **Model comparison**: Always compare multiple models to find the best one

## Next Steps

- Experiment with hyperparameter tuning (GridSearchCV)
- Try other scikit-learn models (SVM, KNN)
- Explore LightGBM and CatBoost alternatives to XGBoost
- Learn about cross-validation for better model evaluation
