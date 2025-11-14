# Demo 1: Statistical Modeling with statsmodels

## Learning Objectives
- Fit and interpret linear regression models using `statsmodels`
- Understand statistical inference (p-values, confidence intervals)
- Compare formula API vs array API
- Generate and analyze realistic datasets
- Visualize model results

## Setup

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import altair as alt
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
```

## Part 1: Load Real Dataset

Let's use the California Housing dataset - a real-world dataset from the 1990 US Census. This dataset contains information about housing prices in California districts and the factors that influence them.

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
print("\nTarget variable (house_value) statistics:")
print(df['house_value'].describe())
```

## Part 2: Formula API - R-like Syntax

The formula API is intuitive and works directly with DataFrames. It's similar to R's modeling syntax.

```python
# Formula API: Simple and intuitive
# Syntax: 'target ~ feature1 + feature2 + ...'
# Let's model house value based on income, house age, and average rooms
model_formula = smf.ols('house_value ~ MedInc + HouseAge + AveRooms', data=df)
results_formula = model_formula.fit()

# Print comprehensive summary
print("=== Formula API Results ===")
print(results_formula.summary())
```

**Key things to look for in the summary:**
- **R-squared**: How well the model fits (0-1, higher is better)
- **Coefficients**: The estimated effect of each variable
- **P-values**: Statistical significance (p < 0.05 is typically significant)
- **Confidence intervals**: Range of plausible values for coefficients

Now let's extract the key statistics programmatically. This is useful when you want to use these values in further analysis or create custom reports.

```python
# Extract key statistics
print("\n=== Key Model Statistics ===")
print(f"R-squared: {results_formula.rsquared:.4f}")
print(f"Adjusted R-squared: {results_formula.rsquared_adj:.4f}")
print(f"\nCoefficients:")
print(results_formula.params)
print(f"\nP-values:")
print(results_formula.pvalues)
print(f"\n95% Confidence Intervals:")
print(results_formula.conf_int())
```

**Understanding these statistics:**
- **R-squared** tells us how much variance in the target is explained by the model
- **Adjusted R-squared** penalizes for model complexity - use this when comparing models with different numbers of predictors
- **Coefficients** show the estimated effect size of each variable
- **P-values** indicate statistical significance - values < 0.05 suggest the variable has a real effect
- **Confidence intervals** give us a range of plausible values for each coefficient

## Part 3: Array API - More Control

The array API gives you more control and is useful when you need to manually construct design matrices.

```python
# Array API: More control over design matrix
# First, prepare the data
y = df['house_value'].values
X = df[['MedInc', 'HouseAge', 'AveRooms']].values

# Add constant (intercept) term
X_with_const = sm.add_constant(X)

# Fit the model
model_array = sm.OLS(y, X_with_const)
results_array = model_array.fit()

print("=== Array API Results ===")
print(results_array.summary())
```

**When to use each API:**
- **Formula API**: Quick exploration, R-like syntax, works with DataFrames
- **Array API**: More control, custom design matrices, integration with NumPy

Both APIs should give identical results. Let's verify this to confirm they're equivalent approaches to the same problem.

```python
# Verify both methods give same results
print("\n=== Comparing Results ===")
print("Formula API coefficients:")
print(results_formula.params)
print("\nArray API coefficients:")
print(results_array.params)
print("\nAre they the same?", np.allclose(results_formula.params.values, results_array.params))
```

As expected, both methods produce identical results. The choice between them depends on your workflow: use the formula API for quick exploration with DataFrames, and the array API when you need more control or are working with NumPy arrays.

## Part 4: Model Diagnostics and Interpretation

Statistical models provide comprehensive diagnostic information beyond just predictions. These diagnostics help us understand model quality and the reliability of our estimates.

```python
# Model diagnostics
print("=== Model Diagnostics ===")
print(f"Number of observations: {results_formula.nobs}")
print(f"Degrees of freedom: {results_formula.df_resid}")
print(f"F-statistic: {results_formula.fvalue:.2f}")
print(f"F-statistic p-value: {results_formula.f_pvalue:.2e}")

# Individual coefficient significance
print("\n=== Coefficient Significance ===")
coef_summary = pd.DataFrame({
    'coefficient': results_formula.params,
    'std_err': results_formula.bse,
    'p_value': results_formula.pvalues,
    'conf_int_lower': results_formula.conf_int()[0],
    'conf_int_upper': results_formula.conf_int()[1]
})
coef_summary['significant'] = coef_summary['p_value'] < 0.05
print(coef_summary)
```

**What these diagnostics tell us:**
- **F-statistic**: Tests whether the model as a whole is significant (better than just using the mean)
- **Degrees of freedom**: Number of observations minus number of parameters - affects statistical tests
- **Standard errors**: Measure of uncertainty in coefficient estimates
- **P-values**: Probability of observing this result if the true coefficient were zero

**Interpreting the results:**
- **MedInc coefficient**: For each unit increase in median income, house value increases (holding other factors constant)
- **HouseAge coefficient**: The effect of house age on value
- **AveRooms coefficient**: The effect of average rooms per household on value
- Check p-values to see which coefficients are statistically significant (p < 0.05)

## Part 5: Making Predictions

Once we have a fitted model, we can make predictions on new data.

```python
# Create new housing data for prediction
new_houses = pd.DataFrame({
    'MedInc': [3.0, 5.0, 8.0],  # Median income
    'HouseAge': [20, 35, 10],   # House age in years
    'AveRooms': [5.0, 6.5, 4.0]  # Average rooms
})

# Make predictions
predictions = results_formula.predict(new_houses)
print("=== Predictions for New Houses ===")
new_houses['predicted_value'] = predictions
print(new_houses)
print("\nNote: Values are in hundreds of thousands of dollars")

# Get prediction intervals (confidence intervals for predictions)
pred_intervals = results_formula.get_prediction(new_houses).conf_int()
new_houses['pred_lower'] = pred_intervals[:, 0]
new_houses['pred_upper'] = pred_intervals[:, 1]
print("\nWith 95% prediction intervals:")
print(new_houses)
```

## Part 6: Visualization with Altair

Let's create informative visualizations of our model results.

```python
# Configure Altair to handle larger datasets
alt.data_transformers.enable('default', max_rows=None)

# Visualize the relationship between variables and house value
# Create a long-form dataset for plotting
plot_data = df.melt(
    id_vars=['house_value'],
    value_vars=['MedInc', 'HouseAge', 'AveRooms'],
    var_name='variable',
    value_name='value'
)

# Create scatter plots with regression lines
base = alt.Chart(plot_data).mark_circle(opacity=0.3).encode(
    x=alt.X('value:Q', title='Variable Value'),
    y=alt.Y('house_value:Q', title='House Value (hundreds of thousands)'),
    color=alt.Color('variable:N', title='Variable')
).properties(
    width=200,
    height=200
)

# Add regression lines
regression = base.transform_regression(
    'value', 'house_value', groupby=['variable']
).mark_line(color='red', strokeWidth=2)

# Combine and facet
chart = (base + regression).facet(
    column=alt.Column('variable:N', title='')
).resolve_scale(
    x='independent',
    y='independent'
)

chart
```

```python
# Visualize coefficient estimates with confidence intervals
coef_plot_data = coef_summary.reset_index()
coef_plot_data = coef_plot_data[coef_plot_data['index'] != 'Intercept']  # Exclude intercept for scale

coef_chart = alt.Chart(coef_plot_data).mark_point(size=100).encode(
    x=alt.X('coefficient:Q', title='Coefficient Estimate'),
    y=alt.Y('index:N', title='Variable', sort='-x'),
    color=alt.condition(
        alt.datum.p_value < 0.05,
        alt.value('green'),
        alt.value('red')
    )
).properties(
    width=400,
    height=200
)

# Add confidence intervals as error bars
error_bars = alt.Chart(coef_plot_data).mark_rule().encode(
    x=alt.X('conf_int_lower:Q', title='Coefficient Estimate'),
    x2='conf_int_upper:Q',
    y='index:N',
    color=alt.condition(
        alt.datum.p_value < 0.05,
        alt.value('green'),
        alt.value('red')
    )
)

(coef_chart + error_bars).resolve_scale(y='shared')
```

## Part 7: Model Comparison

Let's compare models with different sets of predictors to see which performs better.

```python
# Compare different models
models = {
    'Model 1 (income only)': smf.ols('house_value ~ MedInc', data=df),
    'Model 2 (income + age)': smf.ols('house_value ~ MedInc + HouseAge', data=df),
    'Model 3 (income + age + rooms)': smf.ols('house_value ~ MedInc + HouseAge + AveRooms', data=df),
    'Model 4 (all features)': smf.ols('house_value ~ MedInc + HouseAge + AveRooms + AveBedrms + Population + AveOccup + Latitude + Longitude', data=df)
}

# Fit all models and compare
comparison = []
for name, model in models.items():
    results = model.fit()
    comparison.append({
        'model': name,
        'r_squared': results.rsquared,
        'adj_r_squared': results.rsquared_adj,
        'aic': results.aic,
        'bic': results.bic,
        'n_params': len(results.params)
    })

comparison_df = pd.DataFrame(comparison)
print("=== Model Comparison ===")
print(comparison_df.to_string(index=False))

# Visualize model comparison
comparison_long = comparison_df.melt(
    id_vars='model',
    value_vars=['r_squared', 'adj_r_squared'],
    var_name='metric',
    value_name='value'
)

alt.Chart(comparison_long).mark_bar().encode(
    x=alt.X('model:N', title='Model', sort='-y'),
    y=alt.Y('value:Q', title='Metric Value'),
    color='metric:N',
    column='metric:N'
).properties(
    width=150,
    height=200
)
```

**Model selection insights:**
- **R-squared** increases as we add more variables (always true)
- **Adjusted R-squared** accounts for model complexity - use this to compare models
- **AIC/BIC** are information criteria - lower is better
- Model 4 has the best fit, but Model 3 might be preferred for simplicity

## Part 8: Handling Categorical Variables

Real data often includes categorical variables. Let's see how `statsmodels` handles them.

```python
# Create a categorical variable from continuous data for demonstration
# Let's create income categories
df['IncomeCategory'] = pd.cut(df['MedInc'], bins=[0, 2, 4, 6, 10], 
                               labels=['Low', 'Medium', 'High', 'Very High'])

# statsmodels automatically creates dummy variables for categorical variables
model_with_categorical = smf.ols(
    'house_value ~ HouseAge + AveRooms + C(IncomeCategory)',
    data=df
)
results_cat = model_with_categorical.fit()

print("=== Model with Categorical Variable ===")
print(results_cat.summary())

# Check what dummy variables were created
print("\n=== Dummy Variable Encoding ===")
print("Reference category: Low (omitted)")
print("\nCoefficients for income categories:")
income_coefs = results_cat.params[results_cat.params.index.str.contains('IncomeCategory')]
print(income_coefs)
```

**Understanding categorical coefficients:**
- The reference category (Low income) is omitted
- Other coefficients show the difference from the reference
- For example, "Very High" income areas have higher house values than "Low" income areas

## Key Takeaways

1. **Formula API** is intuitive and R-like - great for exploration
2. **Array API** gives more control - useful for custom design matrices
3. **Model summary** provides rich statistical information (R², p-values, confidence intervals)
4. **Statistical inference** helps you understand relationships, not just predict
5. **Model comparison** helps you choose the best model for your needs
6. **Categorical variables** are automatically handled with dummy encoding

## Next Steps

- Try adding interaction terms (e.g., `age * bmi`)
- Experiment with different model specifications
- Explore generalized linear models (GLMs) for non-normal data
- Learn about model diagnostics and assumption checking

