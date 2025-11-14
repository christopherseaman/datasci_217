# Demo 2: Machine Learning with scikit-learn and XGBoost

## Learning Objectives
- Master the scikit-learn fit/predict pattern
- Build and evaluate linear regression and random forest models
- Use XGBoost for gradient boosting
- Understand feature importance
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
import xgboost as xgb
import altair as alt
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
```

## Part 1: Generate Realistic Dataset

Let's create a larger, more realistic dataset for machine learning. We'll simulate housing prices based on various features.

```python
# Generate realistic housing data
n_houses = 5000

np.random.seed(42)
data = {
    'house_id': [f'HOUSE_{i:05d}' for i in range(1, n_houses + 1)],
    'sqft': np.random.normal(2000, 600, n_houses).astype(int),
    'bedrooms': np.random.choice([2, 3, 4, 5], n_houses, p=[0.2, 0.4, 0.3, 0.1]),
    'bathrooms': np.random.choice([1, 1.5, 2, 2.5, 3], n_houses, p=[0.1, 0.2, 0.4, 0.2, 0.1]),
    'age': np.random.exponential(15, n_houses).astype(int),
    'lot_size': np.random.lognormal(7, 0.5, n_houses).astype(int),
    'neighborhood_score': np.random.uniform(1, 10, n_houses),  # School quality, safety, etc.
    'distance_to_city': np.random.exponential(10, n_houses),  # miles
    'has_garage': np.random.choice([0, 1], n_houses, p=[0.3, 0.7]),
    'has_pool': np.random.choice([0, 1], n_houses, p=[0.8, 0.2])
}

df = pd.DataFrame(data)

# Create target: house price (in thousands)
# More complex relationship with some non-linearities
df['price'] = (
    100 +  # Base price
    0.15 * df['sqft'] +  # $150 per sqft
    20 * df['bedrooms'] +  # $20k per bedroom
    25 * df['bathrooms'] +  # $25k per bathroom
    -2 * df['age'] +  # -$2k per year old
    0.05 * df['lot_size'] +  # $50 per sqft of lot
    15 * df['neighborhood_score'] +  # $15k per point
    -3 * df['distance_to_city'] +  # -$3k per mile
    30 * df['has_garage'] +  # $30k for garage
    50 * df['has_pool'] +  # $50k for pool
    # Add some non-linear interactions
    0.001 * df['sqft'] * df['neighborhood_score'] +  # Interaction
    -0.0001 * df['age'] * df['sqft'] +  # Older large houses depreciate more
    np.random.normal(0, 30, n_houses)  # Random noise
)

# Ensure positive prices
df['price'] = df['price'].clip(50, 1000)

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nSummary statistics:")
print(df.describe())
```

## Part 2: Train/Test Split

The golden rule: never evaluate on data the model has seen during training!

```python
# Prepare features and target
feature_cols = ['sqft', 'bedrooms', 'bathrooms', 'age', 'lot_size', 
                'neighborhood_score', 'distance_to_city', 'has_garage', 'has_pool']
X = df[feature_cols]
y = df['price']

# Split into training and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"\nTraining target statistics:")
print(y_train.describe())
print(f"\nTest target statistics:")
print(y_test.describe())
```

## Part 3: Linear Regression with scikit-learn

scikit-learn's API is consistent across all models: create, fit, predict.

```python
# Create and fit linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions
y_train_pred = lr_model.predict(X_train)
y_test_pred = lr_model.predict(X_test)

# Evaluate model
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("=== Linear Regression Results ===")
print(f"Training R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")
print(f"Training RMSE: ${train_rmse:.2f}k")
print(f"Test RMSE: ${test_rmse:.2f}k")
print(f"\nCoefficients:")
coef_df = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': lr_model.coef_
})
print(coef_df)
print(f"\nIntercept: ${lr_model.intercept_:.2f}k")
```

## Part 4: Regularized Linear Models

Regularization helps prevent overfitting by penalizing large coefficients.

```python
# Ridge Regression (L2 regularization)
ridge_model = Ridge(alpha=10.0)  # alpha controls regularization strength
ridge_model.fit(X_train, y_train)
ridge_test_r2 = r2_score(y_test, ridge_model.predict(X_test))
ridge_test_rmse = np.sqrt(mean_squared_error(y_test, ridge_model.predict(X_test)))

# Lasso Regression (L1 regularization - can zero out coefficients)
lasso_model = Lasso(alpha=1.0)
lasso_model.fit(X_train, y_train)
lasso_test_r2 = r2_score(y_test, lasso_model.predict(X_test))
lasso_test_rmse = np.sqrt(mean_squared_error(y_test, lasso_model.predict(X_test)))

print("=== Regularized Models Comparison ===")
print(f"Linear Regression - Test R²: {test_r2:.4f}, RMSE: ${test_rmse:.2f}k")
print(f"Ridge Regression - Test R²: {ridge_test_r2:.4f}, RMSE: ${ridge_test_rmse:.2f}k")
print(f"Lasso Regression - Test R²: {lasso_test_r2:.4f}, RMSE: ${lasso_test_rmse:.2f}k")

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
rf_test_pred = rf_model.predict(X_test)

# Evaluate
rf_train_r2 = r2_score(y_train, rf_train_pred)
rf_test_r2 = r2_score(y_test, rf_test_pred)
rf_train_rmse = np.sqrt(mean_squared_error(y_train, rf_train_pred))
rf_test_rmse = np.sqrt(mean_squared_error(y_test, rf_test_pred))

print("=== Random Forest Results ===")
print(f"Training R²: {rf_train_r2:.4f}")
print(f"Test R²: {rf_test_r2:.4f}")
print(f"Training RMSE: ${rf_train_rmse:.2f}k")
print(f"Test RMSE: ${rf_test_rmse:.2f}k")

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
- Feature importance tells you which variables matter most
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
    eval_set=[(X_test, y_test)],
    verbose=False
)

# Make predictions
xgb_train_pred = xgb_model.predict(X_train)
xgb_test_pred = xgb_model.predict(X_test)

# Evaluate
xgb_train_r2 = r2_score(y_train, xgb_train_pred)
xgb_test_r2 = r2_score(y_test, xgb_test_pred)
xgb_train_rmse = np.sqrt(mean_squared_error(y_train, xgb_train_pred))
xgb_test_rmse = np.sqrt(mean_squared_error(y_test, xgb_test_pred))

print("=== XGBoost Results ===")
print(f"Training R²: {xgb_train_r2:.4f}")
print(f"Test R²: {xgb_test_r2:.4f}")
print(f"Training RMSE: ${xgb_train_rmse:.2f}k")
print(f"Test RMSE: ${xgb_test_rmse:.2f}k")

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
    'Test R²': [test_r2, ridge_test_r2, lasso_test_r2, rf_test_r2, xgb_test_r2],
    'Test RMSE': [test_rmse, ridge_test_rmse, lasso_test_rmse, rf_test_rmse, xgb_test_rmse]
})

print("=== Model Comparison ===")
print(comparison.to_string(index=False))

# Visualize comparison
comparison_long = comparison.melt(
    id_vars='Model',
    value_vars=['Train R²', 'Test R²'],
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

## Part 8: Prediction Visualization

Visualize how well our best model predicts house prices.

```python
# Use XGBoost predictions for visualization
pred_df = pd.DataFrame({
    'actual': y_test.values,
    'predicted': xgb_test_pred,
    'error': y_test.values - xgb_test_pred
})

# Scatter plot: actual vs predicted
scatter = alt.Chart(pred_df).mark_circle(opacity=0.5).encode(
    x=alt.X('actual:Q', title='Actual Price ($k)'),
    y=alt.Y('predicted:Q', title='Predicted Price ($k)'),
    color=alt.Color('error:Q', scale=alt.Scale(scheme='redblue', domainMid=0), 
                    title='Error ($k)')
).properties(
    width=400,
    height=400
)

# Add perfect prediction line (y=x)
perfect_line = alt.Chart(pd.DataFrame({'x': [pred_df['actual'].min(), pred_df['actual'].max()})).mark_line(
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
    x=alt.X('predicted:Q', title='Predicted Price ($k)'),
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

## Part 10: Early Stopping with XGBoost

Early stopping prevents overfitting by stopping training when validation performance stops improving.

```python
# XGBoost with early stopping
xgb_early_stop = xgb.XGBRegressor(
    n_estimators=500,  # Set high, but early stopping will stop earlier
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)

# Fit with early stopping
xgb_early_stop.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=10,  # Stop if no improvement for 10 rounds
    verbose=False
)

# Check how many rounds were actually used
print(f"=== Early Stopping Results ===")
print(f"Best iteration: {xgb_early_stop.best_iteration}")
print(f"Best score: {xgb_early_stop.best_score:.4f}")

# Compare with model without early stopping
xgb_early_pred = xgb_early_stop.predict(X_test)
xgb_early_r2 = r2_score(y_test, xgb_early_pred)
xgb_early_rmse = np.sqrt(mean_squared_error(y_test, xgb_early_pred))

print(f"\nXGBoost (no early stopping) - Test R²: {xgb_test_r2:.4f}, RMSE: ${xgb_test_rmse:.2f}k")
print(f"XGBoost (with early stopping) - Test R²: {xgb_early_r2:.4f}, RMSE: ${xgb_early_rmse:.2f}k")
```

## Key Takeaways

1. **scikit-learn API**: Consistent fit/predict pattern across all models
2. **Train/test split**: Always evaluate on unseen data
3. **Regularization**: Ridge and Lasso help prevent overfitting
4. **Random Forest**: Handles non-linear relationships automatically
5. **XGBoost**: Often the best performer on tabular data
6. **Feature importance**: Understand which variables matter most
7. **Early stopping**: Prevents overfitting in gradient boosting
8. **Model comparison**: Always compare multiple models to find the best one

## Next Steps

- Experiment with hyperparameter tuning (GridSearchCV)
- Try other scikit-learn models (SVM, KNN)
- Explore LightGBM and CatBoost alternatives to XGBoost
- Learn about cross-validation for better model evaluation

