---
marp: true
theme: sqrl
paginate: true
class: invert
---

# Statistical & Machine Learning Methods

- Time series
- `statsmodels`
- `scikit-learn`
- ML packages
  - Keras/TensorFlow
  - PyTorch

<!--
Key points to discuss:
- Time series analysis is crucial for analyzing temporal data patterns
- statsmodels provides comprehensive statistical modeling capabilities
- scikit-learn offers a unified API for machine learning tasks
- Deep learning frameworks (Keras/TensorFlow, PyTorch) enable complex neural network architectures
-->

---

# Time Series Analysis (1/8)

## Introduction to Time Series
- Definition and key components
  ```python
  # Input: data (array-like of numeric values), index (datetime-like array of same length as data)
  # Output: pandas Series with datetime index
  ts = pd.Series(data, index=pd.date_range('2023', periods=12, freq='ME'))
  ```
- Types:
  - Fixed frequency (regular intervals)
  - Irregular frequency (variable intervals)
- Common applications: finance, economics, sensor data

<!--
Key points:
- Time series data represents observations collected over time
- Regular vs irregular intervals affect analysis methods
- Real-world applications span multiple domains
-->

---

# Time Series Analysis (2/8)
## pandas DateTime Objects
```python
# Input: start_date (str in YYYY-MM-DD format), periods (int)
# Output: DatetimeIndex object
dates = pd.date_range('2023-01-01', periods=5)

# Input: date_str (str in YYYY-MM-DD format)
# Output: Timestamp and Period objects
ts = pd.Timestamp('2023-01-01')
period = pd.Period('2023-01', freq='ME')

# Input: timezone_str (str representing timezone)
# Output: timezone-aware Timestamp objects
ts_ny = ts.tz_localize('America/New_York')
ts_utc = ts_ny.tz_convert('UTC')
```

<!--
Key points:
- DateTime handling is crucial for time series analysis
- pandas provides multiple datetime object types
- Timezone awareness is important for global data
-->

---

# Time Series Analysis (3/8)

## Time Series Operations
```python
# Input: freq (str representing frequency e.g., 'ME', 'H'), aggregation function
# Output: resampled DataFrame with new frequency
df.resample('ME').mean()    # Downsample to monthly
df.resample('H').ffill()   # Upsample to hourly

# Input: window (int) size of rolling window
# Output: DataFrame with rolling statistics
df.rolling(window=7).mean()

# Input: periods (int) number of periods to shift
# Output: shifted DataFrame
df.shift(periods=1)        # Forward shift
df.shift(periods=-1)       # Backward shift
```

<!--
Key points:
- Resampling allows changing time series frequency
- Rolling windows enable moving calculations
- Shifting helps analyze lagged relationships
-->

---

# Time Series Analysis (4/8)

## Frequency and Date Ranges
```python
# Input: start (str in YYYY format), freq (str), periods (int)
# Output: DatetimeIndex with specified frequency
pd.date_range('2023', freq='D', periods=365)   # Daily
pd.date_range('2023', freq='B', periods=252)   # Business days
pd.date_range('2023', freq='ME', periods=12)    # Month end
pd.date_range('2023', freq='Q', periods=4)     # Quarter end

# Input: start (str), freq (str), periods (int)
# Output: PeriodIndex with specified frequency
pd.period_range('2023', freq='ME', periods=12)
```

<!--
Key points:
- Different frequency options for various analysis needs
- Business days vs calendar days handling
- Period ranges for fiscal/accounting periods
-->

---

# Time Series Analysis (5/8)

## Moving Window Functions
```python
# Input: window (int), min_periods (int)
# Output: DataFrame with rolling statistics
df.rolling(window=30, min_periods=25).mean()
df.rolling(window=30, min_periods=25).std()

# Input: span (int), min_periods (int)
# Output: DataFrame with exponential weighted statistics
df.ewm(span=30, min_periods=25).mean()

# Input: window (int), custom function
# Output: DataFrame with custom rolling statistics
def custom_stat(x): return x.max() - x.min()
df.rolling(window=30).apply(custom_stat)
```

<!--
Key points:
- Moving windows capture local patterns
- Exponential weighting gives more weight to recent data
- Custom functions enable flexible analysis
-->

---

# Time Series Analysis (6/8)

## Time Series Decomposition
```python
# Input: time series (array-like), model type (str), period (int)
# Output: decomposition object with trend, seasonal, and residual components
result = seasonal_decompose(
    df['value'],
    model='additive',  # or 'multiplicative'
    period=12
)

trend = result.trend
seasonal = result.seasonal
residual = result.resid
```

<!--
Key points:
- Decomposition separates time series into components
- Additive vs multiplicative models for different patterns
- Understanding trend, seasonality, and noise
-->

---

# Time Series Analysis (7/8)

## Time Series Visualization
```python
# Input: DataFrame with datetime index
# Output: matplotlib figure
df.plot(figsize=(12,6))

# Input: DataFrame grouped by time component
# Output: bar plot of seasonal patterns
df.groupby(df.index.month).mean().plot(kind='bar')

# Input: time series data
# Output: autocorrelation plot
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df)
```

<!--
Key points:
- Visualization reveals patterns and anomalies
- Seasonal plots show recurring patterns
- Autocorrelation helps identify time dependencies
-->

---

# Time Series Analysis (8/8)

## Advanced Time Series Models
```python
# Input: time series data, ARIMA order parameters (p,d,q)
# Output: fitted ARIMA model
model = ARIMA(data, order=(1,1,1))
results = model.fit()

# Input: steps (int) for forecast horizon
# Output: point forecasts and confidence intervals
forecast = results.forecast(steps=5)
conf_int = results.get_forecast(steps=5).conf_int()
```

<!--
Key points:
- ARIMA models capture complex time series patterns
- Model parameters affect forecasting accuracy
- Confidence intervals quantify prediction uncertainty
-->

---
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
- Influence measures identify impactful observations
- Leverage indicates potential outliers in predictors
- Cook's distance combines leverage and residuals
- Visual diagnostics help spot problematic points
-->

---

# statsmodels (6/9)

## OLS Panel Data Analysis
```pythoncode 
from statsmodels.regression.linear_model import OLS

# Fixed effects
model_fe = OLS(y, X, entity_effects=True)
fe_results = model_fe.fit()

# Random effects
model_re = OLS(y, X, random_effects=True)
re_results = model_re.fit()
```

<!--
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
- VIF helps detect multicollinearity
- Assumption checking is crucial
- Multiple tests needed for thorough validation
- Automated checks streamline analysis
-->

---
# scikit-learn (1/9)

## Data Preprocessing
```python
# Input: X (array-like of shape (n_samples, n_features))
# Output: standardized features with zero mean and unit variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Input: X_cat (array-like of categorical variables)
# Output: one-hot encoded matrix
enc = OneHotEncoder(sparse=False)
X_encoded = enc.fit_transform(X_cat)
```

<!--
- Preprocessing is crucial for model performance
- Scaling ensures features are comparable
- One-hot encoding handles categorical variables
- Always fit preprocessors on training data only
-->

---

# scikit-learn (2/9)

## Feature Selection
```python
from sklearn.feature_selection import SelectKBest, RFE
from sklearn.ensemble import RandomForestClassifier

# SelectKBest
selector = SelectKBest(k=5)
X_selected = selector.fit_transform(X, y)

# Recursive Feature Elimination
rfe = RFE(RandomForestClassifier(), n_features_to_select=5)
X_rfe = rfe.fit_transform(X, y)
```

<!--
- Feature selection reduces dimensionality
- SelectKBest uses univariate statistical tests
- RFE iteratively removes least important features
- Different methods suit different problems
-->

---

# scikit-learn (3/9)

## Supervised Learning: Classification
```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Models
lr = LogisticRegression()
rf = RandomForestClassifier()
svm = SVC(kernel='rbf')

# Common interface
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

<!--
- Classification predicts categorical outcomes
- Different algorithms suit different problems
- Consistent API across all models
- Model selection depends on data characteristics
-->

---

# scikit-learn (4/9)

## Supervised Learning: Regression
```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Models
lr = LinearRegression()
rf = RandomForestRegressor()
svr = SVR(kernel='rbf')

# Common interface
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

<!--
- Regression predicts continuous outcomes
- Similar interface to classification
- Different algorithms handle nonlinearity differently
- Consider interpretability vs performance
-->

---

# scikit-learn (5/9)

## Unsupervised Learning
```python
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

# Clustering
kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(X)

# Dimensionality reduction
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
```

<!--
- Unsupervised learning finds patterns without labels
- Clustering groups similar data points
- PCA reduces dimensions while preserving variance
- Different methods reveal different patterns
-->

---

# scikit-learn (6/9)

## Model Selection
```python
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# Cross-validation
scores = cross_val_score(model, X, y, cv=5)

# Grid search
params = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}
grid = GridSearchCV(SVC(), params, cv=5)
grid.fit(X, y)
```

<!--
- Cross-validation assesses model generalization
- Grid search optimizes hyperparameters
- Multiple metrics available for evaluation
- Balance between computation and thoroughness
-->

---

# scikit-learn (7/9)

## Model Evaluation
```python
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_squared_error, r2_score

# Classification metrics
print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))

# Regression metrics
print(f"MSE: {mean_squared_error(y_true, y_pred)}")
print(f"R2: {r2_score(y_true, y_pred)}")
```

<!--
- Different metrics suit different problems
- Classification report provides comprehensive view
- Confusion matrix shows error types
- R-squared and MSE common for regression
-->

---

# scikit-learn (8/9)

## Pipeline Construction
```python
from sklearn.pipeline import Pipeline, FeatureUnion

# Create pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(k=5)),
    ('classifier', LogisticRegression())
])

# Feature union
union = FeatureUnion([
    ('numeric', numeric_pipeline),
    ('text', text_pipeline)
])
```

<!--
- Pipelines chain operations together
- Ensures consistent preprocessing
- Feature union combines different features
- Helps prevent data leakage
-->

---

# scikit-learn (9/9)

## Model Persistence
```python
import joblib

# Save model
joblib.dump(model, 'model.joblib')

# Load model
model = joblib.load('model.joblib')

# Update model
model.partial_fit(X_new, y_new)
```

<!--
- Model persistence saves trained models
- Joblib efficient for large NumPy arrays
- Partial fitting allows incremental updates
- Consider version compatibility
-->

