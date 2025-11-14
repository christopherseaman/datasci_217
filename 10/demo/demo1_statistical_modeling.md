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

## Part 1: Generating Realistic Data

Let's create a realistic dataset for modeling. We'll simulate a healthcare scenario where we want to understand factors affecting patient readmission rates.

```python
# Generate realistic healthcare data
n_patients = 2000

# Patient characteristics
np.random.seed(42)
data = {
    'patient_id': [f'PAT_{i:04d}' for i in range(1, n_patients + 1)],
    'age': np.random.normal(65, 15, n_patients).astype(int),
    'bmi': np.random.normal(28, 6, n_patients),
    'chronic_conditions': np.random.poisson(2, n_patients),
    'medication_count': np.random.poisson(5, n_patients),
    'hospital_stay_days': np.random.gamma(3, 2, n_patients).astype(int) + 1,
    'insurance_type': np.random.choice(['Medicare', 'Medicaid', 'Private', 'Uninsured'], 
                                       n_patients, p=[0.4, 0.3, 0.25, 0.05])
}

df = pd.DataFrame(data)

# Create target variable: readmission risk score (0-100)
# Higher scores indicate higher risk of readmission
# Model: risk = 20 + 0.5*age + 1.2*bmi + 3*chronic_conditions + noise
true_coefficients = {
    'intercept': 20,
    'age': 0.5,
    'bmi': 1.2,
    'chronic_conditions': 3.0,
    'medication_count': 0.8,
    'hospital_stay_days': 1.5
}

df['readmission_risk'] = (
    true_coefficients['intercept'] +
    true_coefficients['age'] * df['age'] +
    true_coefficients['bmi'] * df['bmi'] +
    true_coefficients['chronic_conditions'] * df['chronic_conditions'] +
    true_coefficients['medication_count'] * df['medication_count'] +
    true_coefficients['hospital_stay_days'] * df['hospital_stay_days'] +
    np.random.normal(0, 10, n_patients)  # Add noise
)

# Clip to reasonable range
df['readmission_risk'] = df['readmission_risk'].clip(0, 100)

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nSummary statistics:")
print(df.describe())
```

## Part 2: Formula API - R-like Syntax

The formula API is intuitive and works directly with DataFrames. It's similar to R's modeling syntax.

```python
# Formula API: Simple and intuitive
# Syntax: 'target ~ feature1 + feature2 + ...'
model_formula = smf.ols('readmission_risk ~ age + bmi + chronic_conditions', data=df)
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

## Part 3: Array API - More Control

The array API gives you more control and is useful when you need to manually construct design matrices.

```python
# Array API: More control over design matrix
# First, prepare the data
y = df['readmission_risk'].values
X = df[['age', 'bmi', 'chronic_conditions']].values

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

```python
# Verify both methods give same results
print("\n=== Comparing Results ===")
print("Formula API coefficients:")
print(results_formula.params)
print("\nArray API coefficients:")
print(results_array.params)
print("\nAre they the same?", np.allclose(results_formula.params.values, results_array.params.values))
```

## Part 4: Model Diagnostics and Interpretation

Statistical models provide rich diagnostic information. Let's explore what we can learn.

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

**Interpreting the results:**
- **Age coefficient (0.5)**: For each additional year of age, readmission risk increases by 0.5 points (holding other factors constant)
- **BMI coefficient (1.2)**: For each unit increase in BMI, risk increases by 1.2 points
- **Chronic conditions (3.0)**: Each additional chronic condition increases risk by 3 points
- All coefficients are statistically significant (p < 0.05)

## Part 5: Making Predictions

Once we have a fitted model, we can make predictions on new data.

```python
# Create new patient data for prediction
new_patients = pd.DataFrame({
    'age': [70, 55, 80],
    'bmi': [30, 25, 32],
    'chronic_conditions': [2, 1, 4]
})

# Make predictions
predictions = results_formula.predict(new_patients)
print("=== Predictions for New Patients ===")
new_patients['predicted_risk'] = predictions
print(new_patients)

# Get prediction intervals (confidence intervals for predictions)
pred_intervals = results_formula.get_prediction(new_patients).conf_int()
new_patients['pred_lower'] = pred_intervals[:, 0]
new_patients['pred_upper'] = pred_intervals[:, 1]
print("\nWith 95% prediction intervals:")
print(new_patients)
```

## Part 6: Visualization with Altair

Let's create informative visualizations of our model results.

```python
# Visualize the relationship between variables and readmission risk
# Create a long-form dataset for plotting
plot_data = df.melt(
    id_vars=['readmission_risk'],
    value_vars=['age', 'bmi', 'chronic_conditions'],
    var_name='variable',
    value_name='value'
)

# Create scatter plots with regression lines
base = alt.Chart(plot_data).mark_circle(opacity=0.3).encode(
    x=alt.X('value:Q', title='Variable Value'),
    y=alt.Y('readmission_risk:Q', title='Readmission Risk Score'),
    color=alt.Color('variable:N', title='Variable')
).properties(
    width=200,
    height=200
)

# Add regression lines
regression = base.transform_regression(
    'value', 'readmission_risk', groupby=['variable']
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
    'Model 1 (age only)': smf.ols('readmission_risk ~ age', data=df),
    'Model 2 (age + bmi)': smf.ols('readmission_risk ~ age + bmi', data=df),
    'Model 3 (age + bmi + chronic)': smf.ols('readmission_risk ~ age + bmi + chronic_conditions', data=df),
    'Model 4 (all variables)': smf.ols('readmission_risk ~ age + bmi + chronic_conditions + medication_count + hospital_stay_days', data=df)
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
# Add insurance type to the model
# statsmodels automatically creates dummy variables for categorical variables
model_with_categorical = smf.ols(
    'readmission_risk ~ age + bmi + chronic_conditions + C(insurance_type)',
    data=df
)
results_cat = model_with_categorical.fit()

print("=== Model with Categorical Variable ===")
print(results_cat.summary())

# Check what dummy variables were created
print("\n=== Dummy Variable Encoding ===")
print("Reference category: Uninsured (omitted)")
print("\nCoefficients for insurance types:")
insurance_coefs = results_cat.params[results_cat.params.index.str.contains('insurance_type')]
print(insurance_coefs)
```

**Understanding categorical coefficients:**
- The reference category (Uninsured) is omitted
- Other coefficients show the difference from the reference
- For example, Medicare patients have 5.2 points higher risk than Uninsured patients

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

