# Demo 1: Statistical Modeling and Framing a Prediction Problem

## Learning Objectives
- Fit and interpret linear regression models using `statsmodels`
- Understand statistical inference (p-values, confidence intervals)
- Compare formula API vs array API
- Check residuals against fitted values
- Visualize model results
- Frame a prediction problem: target, feature availability, and a chronological split

## Setup

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import altair as alt

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
- **Coefficients**: The estimated association for each variable under the model
- **P-values**: How incompatible data this extreme are with a specified null model; compare them with a pre-specified threshold such as 0.05
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
- **Coefficients** show the estimated association for each variable, conditional on the model and its other predictors
- **P-values** quantify how incompatible data this extreme are with a specified null model; crossing 0.05 is not proof of a real or causal effect
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

Coefficients only mean something if the straight-line form fits. Plot each row's residual (observed minus fitted) against its fitted value: a shapeless cloud around zero is what we want, while a curve or a funnel says the form or the constant-spread assumption is wrong.

```python
# Residuals versus fitted values
diagnostics = pd.DataFrame({
    'fitted': results_formula.fittedvalues,
    'residual': results_formula.resid,
})
print(diagnostics.describe().round(3))
print(f"\nFitted values below 0 (impossible house values): {(diagnostics['fitted'] < 0).sum()}")

# Plot a sample so the chart stays light
resid_sample = diagnostics.sample(2000, random_state=42)

resid_points = alt.Chart(resid_sample).mark_circle(opacity=0.3).encode(
    x=alt.X('fitted:Q', title='Fitted house value (hundreds of thousands)'),
    y=alt.Y('residual:Q', title='Residual (observed - fitted)')
).properties(width=400, height=300)

zero_rule = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(
    color='gray', strokeDash=[5, 5]
).encode(y='y:Q')

resid_points + zero_rule
```

**What the checkpoint shows:** residuals average 0 by construction, but the fitted values run from about -2.4 to 7.3 while real house values are capped at 5.0. The straight line predicts impossible values at both ends, which is the kind of pattern a residual plot is meant to expose.

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

# Get intervals for individual future observations, not just mean responses.
# ``obs=True`` includes the residual variation in a new house value.
pred_intervals = results_formula.get_prediction(new_houses).conf_int(obs=True)
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

## Part 9: Framing a Prediction Problem

Every model so far used all 20,640 census rows at once, which is the right thing to do when the question is _how_ income relates to house value. A prediction question - "what will this unit's value be next time we measure it?" - is judged on rows the model has never seen, and the census table has no time order to split on.

So this part switches to a small clinic table with repeat visits, like the one in the lecture: 30 patients, eight weekly visits each. The target is the patient's **next** visit SBP, so `shift(-1)` pulls each patient's next reading back onto the current row, and the target is measured seven days after the features.

```python
clinic_rng = np.random.default_rng(217)
visit_dates = pd.date_range('2026-01-05', periods=8, freq='W-MON')

records = []
for patient_id in range(1, 31):
    age = int(clinic_rng.integers(35, 80))
    sbp = float(clinic_rng.normal(132 + 0.2 * age, 8))
    for visit_date in visit_dates:
        sbp = 0.7 * sbp + 0.3 * clinic_rng.normal(132 + 0.2 * age, 8)
        records.append({
            'patient_id': patient_id,
            'visit_date': visit_date,
            'age': age,
            'sbp_today': round(sbp, 1),
            'a1c_result': round(float(clinic_rng.normal(6.4, 0.9)), 1),  # lab reports the next day
        })

visits = pd.DataFrame(records).sort_values(['patient_id', 'visit_date'])
visits['sbp_next_visit'] = visits.groupby('patient_id')['sbp_today'].shift(-1)
visits['target_date'] = visits['visit_date'] + pd.Timedelta(days=7)
visits = visits.dropna(subset=['sbp_next_visit']).reset_index(drop=True)

print(f"Prediction unit: one visit. Target: sbp_next_visit, measured on target_date.")
print(f"Rows with a target: {len(visits)}")
print(visits.head(3)[['patient_id', 'visit_date', 'sbp_today', 'sbp_next_visit', 'target_date']])
```

Expect 210 rows: 30 patients times eight visits, minus each patient's last visit, which has no next reading.

Now audit every candidate feature. The prediction is made at the end of the visit, so anything that becomes known afterwards is leakage.

```python
candidates = pd.DataFrame({
    'candidate_feature': ['age', 'sbp_today', 'a1c_result'],
    'becomes_known': ['Before the visit', 'During the visit', 'Lab reports next day'],
    'hours_after_visit': [0, 0, 24],
})
candidates['available'] = candidates['hours_after_visit'] <= 0
candidates['decision'] = np.where(candidates['available'], 'keep', 'exclude (leakage)')
print(candidates)

features = candidates.loc[candidates['available'], 'candidate_feature'].tolist()
print(f"\nFeatures the model may use: {features}")
```

`a1c_result` is excluded: it is measurable, informative, and unavailable at prediction time, which is exactly the trap. Next, split on the **target** date, not the visit date, so no training outcome is measured during the validation weeks.

```python
train = visits[visits['target_date'] < '2026-02-09']
valid = visits[(visits['target_date'] >= '2026-02-09') & (visits['target_date'] < '2026-02-23')]
test = visits[visits['target_date'] >= '2026-02-23']

for name, part in [('train', train), ('valid', valid), ('test', test)]:
    print(f"{name}: {len(part):3d} rows, targets "
          f"{part['target_date'].min().date()} to {part['target_date'].max().date()}")
```

```text
train: 120 rows, targets 2026-01-12 to 2026-02-02
valid:  60 rows, targets 2026-02-09 to 2026-02-16
test:  30 rows, targets 2026-02-23 to 2026-02-23
```

The three target-date ranges do not overlap. Fit on the training rows only, then read the validation rows with the same prediction intervals from Part 5.

```python
honest_fit = smf.ols('sbp_next_visit ~ age + sbp_today', data=train).fit()
print(honest_fit.params.round(3))
print(honest_fit.conf_int().round(2))

valid_pred = honest_fit.get_prediction(valid).summary_frame(alpha=0.05)
check = valid[['patient_id', 'target_date', 'sbp_next_visit']].head(3).copy()
check['predicted'] = valid_pred['mean'].head(3).values.round(1)
check['pi_lower'] = valid_pred['obs_ci_lower'].head(3).values.round(1)
check['pi_upper'] = valid_pred['obs_ci_upper'].head(3).values.round(1)
print(check)
```

The `sbp_today` coefficient is about 0.67, so a patient's reading carries most of the way to the next one, and each prediction interval spans roughly 10 mmHg. Demo 2 goes back to the California Housing table, where the rows have no time order, and runs the rest of the prediction workflow with scikit-learn: a random split, a baseline to beat, pipelines, and error metrics.

## Key Takeaways

1. **Formula API** is intuitive and R-like - great for exploration
2. **Array API** gives more control - useful for custom design matrices
3. **Model summary** provides rich statistical information (R², p-values, confidence intervals)
4. **Residual plots** check the straight-line form before you trust a coefficient
5. **Statistical inference** helps you understand relationships, not just predict
6. **Model comparison** with adjusted R², AIC, and BIC weighs fit against complexity
7. **Categorical variables** are automatically handled with dummy encoding
8. **Prediction framing** comes before any prediction model: name the target and its time, exclude features that arrive too late, and split chronologically

## Next Steps

- Try adding interaction terms (e.g., `age * bmi`)
- Experiment with different model specifications
- Explore generalized linear models (GLMs) for non-normal data
- Learn about model diagnostics and assumption checking
