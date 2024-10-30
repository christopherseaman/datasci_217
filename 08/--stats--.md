# statsmodels (1/9)

## Linear Regression Basics
```python
import statsmodels.api as sm

# Input: X (array-like, independent variables), y (array-like, dependent variable)
# Note: X should be a matrix of independent variables (features)
#       Each column represents a different predictor variable
#       Rows represent observations
X = sm.add_constant(X)  # Add intercept column to X
model = sm.OLS(y, X)
results = model.fit()

# Key outputs
print(results.params)        # Coefficients
print(results.rsquared)     # R-squared
print(results.pvalues)      # P-values
```

<!--
Talking points:
- statsmodels provides comprehensive statistical modeling tools
- OLS is the foundation of linear regression analysis
- Adding a constant term is crucial for proper model fitting
- Model results provide multiple statistical measures
-->

---

# statsmodels (2/9)

## Model Diagnostics
```python
# Input: model results object
# Output: various diagnostic statistics
print(results.rsquared)
print(results.rsquared_adj)

# Input: model results object
# Output: p-values for each coefficient
print(results.pvalues)

# Input: residuals (array-like), X matrix
# Output: heteroskedasticity test statistics
residuals = results.resid
sm.stats.diagnostic.het_breuschpagan(residuals, X)
```

<!--
Talking points:
- Model diagnostics help validate assumptions
- R-squared vs adjusted R-squared tradeoffs
- P-values indicate statistical significance
- Heteroskedasticity testing is crucial for valid inference
-->

---

# statsmodels (3/9)

## Generalized Linear Models

```python
import statsmodels.api as sm

# GLM specification
model = sm.GLM(y, X, family=sm.families.Binomial())
results = model.fit()

# Link functions
sm.families.Gaussian(link=sm.families.links.log)
sm.families.Poisson(link=sm.families.links.log)
```

<!--
Talking points:
- GLMs extend linear regression to non-normal distributions
- Different families handle different response types
- Link functions connect linear predictor to response
- Common applications in binary and count data
-->

---

# statsmodels (4/9)

## Statistical Tests
```python
# T-test
from scipy import stats
t_stat, p_val = stats.ttest_ind(group1, group2)

# ANOVA
from statsmodels.stats.anova import anova_lm
anova_table = anova_lm(fitted_model, typ=2)

# Chi-square
chi2, p_val = stats.chi2_contingency(contingency_table)
```

<!--
Talking points:
- Statistical tests help validate hypotheses
- T-tests compare means between groups
- ANOVA examines variance between groups
- Chi-square tests analyze categorical relationships
-->

---

# statsmodels (5/9)

## Regression Diagnostics
```python
# Influence statistics
influence = results.get_influence()
leverage = influence.hat_matrix_diag
cooks = influence.cooks_distance[0]

# Outlier detection
student_resid = results.outlier_test()

# Leverage plots
sm.graphics.influence_plot(results)
```

<!--
Talking points:
- Influence measures identify impactful observations
- Leverage indicates potential outliers in predictors
- Cook's distance combines leverage and residuals
- Visual diagnostics help spot problematic points
-->

---

# statsmodels (6/9)

## Panel Data Analysis
```python
from statsmodels.regression.linear_model import PanelOLS

# Fixed effects
model_fe = PanelOLS(y, X, entity_effects=True)
fe_results = model_fe.fit()

# Random effects
model_re = PanelOLS(y, X, random_effects=True)
re_results = model_re.fit()
```

<!--
Talking points:
- Panel data combines cross-sectional and time series
- Fixed effects control for time-invariant factors
- Random effects assume random individual differences
- Choice between models depends on Hausman test
-->

---

# statsmodels (7/9)

## Robust Regression
```python
# Robust linear model
model = sm.RLM(y, X)
results = model.fit()

# Robust standard errors
results = model.fit(cov_type='HC3')

# M-estimators
results = model.fit(method='huber')
```

<!--
Talking points:
- Robust methods handle outliers and violations
- Different types of robust standard errors
- M-estimators provide resistance to outliers
- Trade-off between efficiency and robustness
-->

---

# statsmodels (8/9)

## Model Output & Visualization
```python
# Summary tables
print(results.summary())

# Diagnostic plots
fig = sm.graphics.plot_regress_exog(results, 'x1')
fig = sm.graphics.plot_fit(results, 'x1')

# Results extraction
conf_int = results.conf_int()
predicted = results.predict(X_new)
```

<!--
Talking points:
- Summary tables provide comprehensive results
- Visual diagnostics help assess assumptions
- Confidence intervals quantify uncertainty
- Prediction helps validate model performance
-->

---

# statsmodels (9/9)

## Advanced Applications
```python
# Model comparison
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Common pitfalls check
def check_assumptions(results):
    print("Normality:", stats.jarque_bera(results.resid)[1])
    print("Heteroskedasticity:", stats.het_breuschpagan(results.resid, X)[1])
    print("Autocorrelation:", stats.durbin_watson(results.resid))
```

<!--
Talking points:
- VIF helps detect multicollinearity
- Assumption checking is crucial
- Multiple tests needed for thorough validation
- Automated checks streamline analysis
-->

---
